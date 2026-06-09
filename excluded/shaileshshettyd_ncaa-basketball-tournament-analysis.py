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


import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import log_loss, brier_score_loss, mean_squared_error, roc_curve, auc
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
import optuna
import warnings
warnings.filterwarnings("ignore")


class MarchMadnessPredictor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = {}
        self.teams = None
        self.games = None
        self.season_results = None
        self.tourney_results = None
        self.seeds = None
        self.submission = None
        self.feature_cols = None
        self.imputer = SimpleImputer(strategy='mean')
        self.scaler = StandardScaler()
        
        # Models
        self.main_model = None
        self.calibration_model = None


def load_data(self):
    """Load and preprocess all necessary datasets"""
    print("Loading data files...")
    
    # Load all CSV files from the data directory
    if '**' in self.data_path:
        files = glob.glob(self.data_path)
        self.data = {p.split('/')[-1].split('.')[0]: pd.read_csv(p, encoding='latin-1') for p in files}
    else:
        for filename in os.listdir(self.data_path):
            if filename.endswith('.csv'):
                file_path = os.path.join(self.data_path, filename)
                self.data[filename.split('.')[0]] = pd.read_csv(file_path)
    
    # Load teams and team spellings
    print("Processing teams data...")
    teams = pd.concat([self.data.get('MTeams', pd.DataFrame()), 
                       self.data.get('WTeams', pd.DataFrame())])
    
    teams_spelling = pd.concat([self.data.get('MTeamSpellings', pd.DataFrame()), 
                               self.data.get('WTeamSpellings', pd.DataFrame())])
    
    if not teams_spelling.empty:
        teams_spelling = teams_spelling.groupby(by='TeamID', as_index=False)['TeamNameSpelling'].count()
        teams_spelling.columns = ['TeamID', 'TeamNameCount']
        self.teams = pd.merge(teams, teams_spelling, how='left', on=['TeamID'])
    else:
        self.teams = teams
    
    # Load season results
    print("Processing season results...")
    self.season_results = pd.concat([
        self.data.get('MRegularSeasonCompactResults', pd.DataFrame()),
        self.data.get('WRegularSeasonCompactResults', pd.DataFrame())
    ], ignore_index=True)
    
    if not self.season_results.empty:
        self.season_results.drop(['NumOT', 'WLoc'] if 'NumOT' in self.season_results.columns else [], 
                                axis=1, inplace=True)
        self.season_results['ScoreGap'] = self.season_results['WScore'] - self.season_results['LScore']
    
    # Load tournament results
    print("Processing tournament results...")
    self.tourney_results = pd.concat([
        self.data.get('MNCAATourneyCompactResults', pd.DataFrame()),
        self.data.get('WNCAATourneyCompactResults', pd.DataFrame())
    ], ignore_index=True)
    
    if not self.tourney_results.empty:
        self.tourney_results.drop(['NumOT', 'WLoc'] if 'NumOT' in self.tourney_results.columns else [], 
                                 axis=1, inplace=True)
    
    # Load seeds
    print("Processing seeds data...")
    seeds_df = pd.concat([
        self.data.get('MNCAATourneySeeds', pd.DataFrame()),
        self.data.get('WNCAATourneySeeds', pd.DataFrame())
    ], ignore_index=True)
    
    if not seeds_df.empty:
        self.seeds = {'_'.join(map(str, [int(k1), k2])): int(v[1:3]) 
                      for k1, v, k2 in seeds_df[['Season', 'Seed', 'TeamID']].values}
    
    # Load submission template
    self.submission = self.data.get('SampleSubmissionStage1', pd.DataFrame())
    
    # Process detailed results for advanced features
    self._process_detailed_results()
    
    print("Data loading complete.")


def _process_detailed_results(self):
    """Process detailed results to extract additional features"""
    season_detailed = pd.concat([
        self.data.get('MRegularSeasonDetailedResults', pd.DataFrame()),
        self.data.get('WRegularSeasonDetailedResults', pd.DataFrame())
    ], ignore_index=True)
    
    tourney_detailed = pd.concat([
        self.data.get('MNCAATourneyDetailedResults', pd.DataFrame()),
        self.data.get('WNCAATourneyDetailedResults', pd.DataFrame())
    ], ignore_index=True)
    
    if not season_detailed.empty:
        season_detailed['ST'] = 'S'  # Season
    
    if not tourney_detailed.empty:
        tourney_detailed['ST'] = 'T'  # Tournament
    
    # Combine all game data
    self.games = pd.concat((season_detailed, tourney_detailed), axis=0, ignore_index=True)
    
    if not self.games.empty:
        # Create game IDs and map locations
        if 'WLoc' in self.games.columns:
            self.games['WLoc'] = self.games['WLoc'].map({'A': 1, 'H': 2, 'N': 3})
        
        # Create game identifiers
        self.games['ID'] = self.games.apply(
            lambda r: '_'.join(map(str, [r['Season']] + sorted([r['WTeamID'], r['LTeamID']]))), axis=1)
        self.games['IDTeams'] = self.games.apply(
            lambda r: '_'.join(map(str, sorted([r['WTeamID'], r['LTeamID']]))), axis=1)
        self.games['Team1'] = self.games.apply(
            lambda r: sorted([r['WTeamID'], r['LTeamID']])[0], axis=1)
        self.games['Team2'] = self.games.apply(
            lambda r: sorted([r['WTeamID'], r['LTeamID']])[1], axis=1)
        
        # Create team identifiers
        self.games['IDTeam1'] = self.games.apply(
            lambda r: '_'.join(map(str, [r['Season'], r['Team1']])), axis=1)
        self.games['IDTeam2'] = self.games.apply(
            lambda r: '_'.join(map(str, [r['Season'], r['Team2']])), axis=1)
        
        # Add seed information
        if self.seeds:
            self.games['Team1Seed'] = self.games['IDTeam1'].map(self.seeds).fillna(0)
            self.games['Team2Seed'] = self.games['IDTeam2'].map(self.seeds).fillna(0)
            self.games['SeedDiff'] = self.games['Team1Seed'] - self.games['Team2Seed']
        
        # Create target variables
        self.games['ScoreDiff'] = self.games['WScore'] - self.games['LScore']
        self.games['Pred'] = self.games.apply(
            lambda r: 1.0 if sorted([r['WTeamID'], r['LTeamID']])[0] == r['WTeamID'] else 0.0, axis=1)
        self.games['ScoreDiffNorm'] = self.games.apply(
            lambda r: r['ScoreDiff'] * -1 if r['Pred'] == 0.0 else r['ScoreDiff'], axis=1)
        
        # Fill missing values
        self.games = self.games.fillna(-1)
        
        # Keep only tournament games for training
        self.games = self.games[self.games['ST'] == 'T']


def _create_team_stats(self):
    """Create team performance statistics"""
    print("Creating team statistics...")
    
    if self.season_results.empty:
        print("Warning: No season results available for team statistics")
        return
    
    # Team wins
    num_wins = self.season_results.groupby(['Season', 'WTeamID']).count()
    num_wins = num_wins.reset_index()[['Season', 'WTeamID', 'DayNum']].rename(
        columns={"DayNum": "NumWins", "WTeamID": "TeamID"})
    
    # Team losses
    num_losses = self.season_results.groupby(['Season', 'LTeamID']).count()
    num_losses = num_losses.reset_index()[['Season', 'LTeamID', 'DayNum']].rename(
        columns={"DayNum": "NumLosses", "LTeamID": "TeamID"})
    
    # Score gaps for wins
    gap_wins = self.season_results.groupby(['Season', 'WTeamID']).mean().reset_index()
    gap_wins = gap_wins[['Season', 'WTeamID', 'ScoreGap']].rename(
        columns={"ScoreGap": "GapWins", "WTeamID": "TeamID"})
    
    # Score gaps for losses
    gap_losses = self.season_results.groupby(['Season', 'LTeamID']).mean().reset_index()
    gap_losses = gap_losses[['Season', 'LTeamID', 'ScoreGap']].rename(
        columns={"ScoreGap": "GapLosses", "LTeamID": "TeamID"})
    
    # Create a base dataframe with all team-season combinations
    df_teams_w = self.season_results.groupby(['Season', 'WTeamID']).count().reset_index()[
        ['Season', 'WTeamID']].rename(columns={"WTeamID": "TeamID"})
    df_teams_l = self.season_results.groupby(['Season', 'LTeamID']).count().reset_index()[
        ['Season', 'LTeamID']].rename(columns={"LTeamID": "TeamID"})
    
    df_features = pd.concat([df_teams_w, df_teams_l], axis=0).drop_duplicates().sort_values(
        ['Season', 'TeamID']).reset_index(drop=True)
    
    # Merge all stats
    df_features = df_features.merge(num_wins, on=['Season', 'TeamID'], how='left')
    df_features = df_features.merge(num_losses, on=['Season', 'TeamID'], how='left')
    df_features = df_features.merge(gap_wins, on=['Season', 'TeamID'], how='left')
    df_features = df_features.merge(gap_losses, on=['Season', 'TeamID'], how='left')
    
    # Fill missing values
    df_features.fillna(0, inplace=True)
    
    # Compute derived features
    df_features['WinRatio'] = df_features['NumWins'] / (df_features['NumWins'] + df_features['NumLosses'])
    df_features['GapAvg'] = (
        (df_features['NumWins'] * df_features['GapWins'] - 
         df_features['NumLosses'] * df_features['GapLosses']) / 
        (df_features['NumWins'] + df_features['NumLosses'])
    )
    
    # Add to games data
    if not self.games.empty:
        # Add stats for the first team
        self.games = pd.merge(
            self.games,
            df_features,
            how='left',
            left_on=['Season', 'Team1'],
            right_on=['Season', 'TeamID']
        ).rename(columns={
            'NumWins': 'Team1NumWins',
            'NumLosses': 'Team1NumLosses',
            'GapWins': 'Team1GapWins',
            'GapLosses': 'Team1GapLosses',
            'WinRatio': 'Team1WinRatio',
            'GapAvg': 'Team1GapAvg',
        }).drop(columns='TeamID', axis=1)
        
        # Add stats for the second team
        self.games = pd.merge(
            self.games,
            df_features,
            how='left',
            left_on=['Season', 'Team2'],
            right_on=['Season', 'TeamID']
        ).rename(columns={
            'NumWins': 'Team2NumWins',
            'NumLosses': 'Team2NumLosses',
            'GapWins': 'Team2GapWins',
            'GapLosses': 'Team2GapLosses',
            'WinRatio': 'Team2WinRatio',
            'GapAvg': 'Team2GapAvg',
        }).drop(columns='TeamID', axis=1)
        
        # Create feature differences
        self.games['WinRatioDiff'] = self.games['Team1WinRatio'] - self.games['Team2WinRatio']
        self.games['GapAvgDiff'] = self.games['Team1GapAvg'] - self.games['Team2GapAvg']
        
        # Fill missing values
        self.games.fillna(0, inplace=True)


def prepare_features(self):
    """Prepare features for model training"""
    print("Preparing features...")
    
    # Create team statistics
    self._create_team_stats()
    
    # Generate aggregated statistics if detailed data is available
    if not self.games.empty and 'WFGM' in self.games.columns:
        stat_cols = ['NumOT', 'WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WOR', 'WDR', 
                     'WAst', 'WTO', 'WStl', 'WBlk', 'WPF', 'LFGM', 'LFGA', 'LFGM3', 'LFGA3', 
                     'LFTM', 'LFTA', 'LOR', 'LDR', 'LAst', 'LTO', 'LStl', 'LBlk', 'LPF']
        
        stat_aggs = ['sum', 'mean', 'median', 'max', 'min', 'std']
        
        # Aggregate statistics by team pairs
        gb = self.games.groupby(by=['IDTeams']).agg({k: stat_aggs for k in stat_cols if k in self.games.columns})
        gb.columns = [''.join(c) + '_c_score' for c in gb.columns]
        gb = gb.reset_index()
        
        # Add aggregated statistics to games data
        self.games = pd.merge(self.games, gb, how='left', left_on='IDTeams', right_on='IDTeams_c_score')
        
        # Add to submission data
        if not self.submission.empty:
            self.submission['IDTeams'] = self.submission.apply(
                lambda r: '_'.join(map(str, [r['Team1'], r['Team2']])), axis=1)
            self.submission = pd.merge(self.submission, gb, how='left', 
                                      left_on='IDTeams', right_on='IDTeams_c_score')
    
    # Define feature columns for the model
    if not self.games.empty:
        # Extract feature columns excluding metadata and target columns
        exclude_cols = ['ID', 'DayNum', 'ST', 'Team1', 'Team2', 'IDTeams', 'IDTeam1', 'IDTeam2', 
                       'WTeamID', 'WScore', 'LTeamID', 'LScore', 'NumOT', 'Pred', 'ScoreDiff', 
                       'ScoreDiffNorm', 'WLoc']
        
        # Add any other columns that might be in the DataFrame but not needed as features
        if 'IDTeams_c_score' in self.games.columns:
            exclude_cols.append('IDTeams_c_score')
        
        # Get all columns except excluded ones
        self.feature_cols = [c for c in self.games.columns if c not in exclude_cols]
        
        print(f"Selected {len(self.feature_cols)} features for modeling")


def create_models(self, model_type='random_forest'):
    """Create the main model and calibration model"""
    print(f"Creating models (type: {model_type})...")
    
    if model_type.lower() == 'catboost':
        # CatBoost model from first notebook
        self.main_model = CatBoostRegressor(
            iterations=761,
            learning_rate=0.00848,
            depth=10,
            l2_leaf_reg=0.0168,
            bagging_temperature=1.934,
            random_strength=1.54e-7,
            task_type='CPU',
            verbose=False
        )
        
        # Calibration model
        self.calibration_model = RandomForestRegressor(
            n_estimators=100, 
            random_state=42, 
            n_jobs=-1, 
            max_depth=10
        )
    else:
        # Random Forest model from second notebook
        self.main_model = RandomForestRegressor(
            n_estimators=235,
            random_state=42,
            max_depth=15,
            min_samples_split=2,
            max_features='sqrt',
            n_jobs=-1
        )
        
        # Calibration model
        self.calibration_model = RandomForestRegressor(
            n_estimators=100, 
            random_state=42, 
            n_jobs=-1, 
            max_depth=10
        )


def hyperparameter_optimization(self, n_trials=5):
    """Optimize model hyperparameters using Optuna"""
    print(f"Starting hyperparameter optimization with {n_trials} trials...")
    
    if self.games.empty or not self.feature_cols:
        print("Error: No game data or features available for optimization")
        return
    
    X = self.games[self.feature_cols].fillna(0)
    y = self.games['Pred']
    
    def objective(trial):
        # Parameters for CatBoost
        params = {
            'iterations': trial.suggest_int('iterations', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True),
            'depth': trial.suggest_int('depth', 3, 12),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 100.0, log=True),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 2.5),
            'random_strength': trial.suggest_float('random_strength', 1e-8, 10.0, log=True),
            'task_type': 'CPU',
            'verbose': False
        }
        
        # Create model with trial parameters
        model = CatBoostRegressor(**params)
        
        # Cross-validation
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)
            
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Predict and clip
            preds = model.predict(X_val_scaled).clip(0.001, 0.999)
            
            # Calculate Brier score
            score = brier_score_loss(y_val, preds)
            scores.append(score)
        
        return np.mean(scores)
    
    # Create and optimize study
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    
    print("Best parameters:", study.best_params)
    
    # Update main model with best parameters
    self.main_model = CatBoostRegressor(**study.best_params)
    
    return study.best_params


def train_model(self):
    """Train the main model and calibration model"""
    print("Training models...")
    
    if self.games.empty or not self.feature_cols:
        print("Error: No game data or features available for training")
        return
    
    # Prepare features and target
    X = self.games[self.feature_cols].fillna(0)
    X_imputed = self.imputer.fit_transform(X)
    X_scaled = self.scaler.fit_transform(X_imputed)
    y = self.games['Pred']
    
    # Split data for training and calibration
    X_train, X_cal, y_train, y_cal = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Train main model
    self.main_model.fit(X_train, y_train)
    
    # Generate predictions for calibration
    train_preds = self.main_model.predict(X_train).clip(0.001, 0.999)
    cal_preds = self.main_model.predict(X_cal).clip(0.001, 0.999)
    
    # Train calibration model
    self.calibration_model.fit(cal_preds.reshape(-1, 1), y_cal)
    
    # Apply calibration to training predictions
    train_preds_calibrated = self.calibration_model.predict(train_preds.reshape(-1, 1)).clip(0.001, 0.999)
    
    # Calculate metrics
    print(f'Log Loss (Train): {log_loss(y_train, train_preds_calibrated):.4f}')
    print(f'Brier Score (Train): {brier_score_loss(y_train, train_preds_calibrated):.4f}')
    print(f'MSE (Train): {mean_squared_error(y_train, train_preds_calibrated):.4f}')
    
    # Cross-validation
    self._cross_validate(X_scaled, y)
    
    # Plot feature importance
    self._plot_feature_importance()
    
    # Plot calibration curve
    self._plot_calibration_curve(y_cal, cal_preds)


def _cross_validate(self, X, y, n_splits=5):
    """Perform cross-validation"""
    print(f"Performing {n_splits}-fold cross-validation...")
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_brier_scores = []
    cv_logloss_scores = []
    
    for train_index, val_index in kf.split(X):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        
        # Train main model
        self.main_model.fit(X_train, y_train)
        val_preds = self.main_model.predict(X_val).clip(0.001, 0.999)
        
        # Train calibration model
        self.calibration_model.fit(val_preds.reshape(-1, 1), y_val)
        val_preds_calibrated = self.calibration_model.predict(val_preds.reshape(-1, 1)).clip(0.001, 0.999)
        
        # Calculate metrics
        brier = brier_score_loss(y_val, val_preds_calibrated)
        logloss = log_loss(y_val, val_preds_calibrated)
        
        cv_brier_scores.append(brier)
        cv_logloss_scores.append(logloss)
    
    print(f'Cross-validated Brier Score: {np.mean(cv_brier_scores):.4f}')
    print(f'Cross-validated Log Loss: {np.mean(cv_logloss_scores):.4f}')


def predict_submission(self, output_file='submission.csv'):
    """Generate predictions for the submission file"""
    print("Generating predictions for submission...")
    
    if self.submission.empty or not self.feature_cols:
        print("Error: No submission data or features available for prediction")
        return
    
    # Prepare features
    sub_features = [col for col in self.feature_cols if col in self.submission.columns]
    sub_X = self.submission[sub_features].fillna(0)
    sub_X_imputed = self.imputer.transform(sub_X)
    sub_X_scaled = self.scaler.transform(sub_X_imputed)
    
    # Generate predictions
    preds = self.main_model.predict(sub_X_scaled).clip(0.001, 0.999)
    
    # Apply calibration
    preds_calibrated = self.calibration_model.predict(preds.reshape(-1, 1)).clip(0.001, 0.999)
    
    # Create submission
    self.submission['Pred'] = preds_calibrated
    self.submission[['ID', 'Pred']].to_csv(output_file, index=False)
    
    print(f"Submission file saved to {output_file}")
    
    # Plot distribution of predictions
    self._plot_prediction_distribution(preds_calibrated)


def add_tournament_features(self):
    """Add tournament-specific features like historical performance"""
    print("Adding tournament-specific features...")
    
    if 'MNCAATourneyCompactResults' not in self.data or 'WNCAATourneyCompactResults' not in self.data:
        print("Warning: Tournament data not available for tournament features")
        return
    
    # Combine men's and women's tournament data
    tourney_results = pd.concat([
        self.data['MNCAATourneyCompactResults'], 
        self.data['WNCAATourneyCompactResults']
    ], ignore_index=True)
    
    # Create tournament experience feature (number of appearances)
    exp_w = tourney_results.groupby(['Season', 'WTeamID']).size().reset_index()
    exp_w.columns = ['Season', 'TeamID', 'TourneyGames']
    
    exp_l = tourney_results.groupby(['Season', 'LTeamID']).size().reset_index()
    exp_l.columns = ['Season', 'TeamID', 'TourneyGames']
    
    # Combine win/loss appearances
    tourney_exp = pd.concat([exp_w, exp_l], ignore_index=True)
    tourney_exp = tourney_exp.groupby(['Season', 'TeamID']).sum().reset_index()
    
    # Calculate cumulative tournament experience
    all_teams = pd.concat([
        tourney_results['WTeamID'].rename('TeamID'),
        tourney_results['LTeamID'].rename('TeamID')
    ]).unique()
    
    all_seasons = tourney_results['Season'].unique()
    
    # Create DataFrame with all team-season combinations
    teams_seasons = pd.DataFrame([
        (season, team) for season in sorted(all_seasons) for team in all_teams
    ], columns=['Season', 'TeamID'])
    
    # Merge with experience data
    teams_seasons = pd.merge(teams_seasons, tourney_exp, on=['Season', 'TeamID'], how='left')
    teams_seasons['TourneyGames'] = teams_seasons['TourneyGames'].fillna(0)
    
    # Calculate prior tournament experience
    teams_seasons = teams_seasons.sort_values(['TeamID', 'Season'])
    teams_seasons['PriorExp'] = teams_seasons.groupby('TeamID')['TourneyGames'].transform(
        lambda x: x.shift(1).fillna(0).cumsum())
    
    # Add features to games data
    if not self.games.empty:
        # Add team 1 experience
        self.games = pd.merge(
            self.games,
            teams_seasons[['Season', 'TeamID', 'PriorExp']],
            how='left',
            left_on=['Season', 'Team1'],
            right_on=['Season', 'TeamID']
        ).rename(columns={'PriorExp': 'Team1TourneyExp'}).drop(columns='TeamID')
        
        # Add team 2 experience
        self.games = pd.merge(
            self.games,
            teams_seasons[['Season', 'TeamID', 'PriorExp']],
            how='left',
            left_on=['Season', 'Team2'],
            right_on=['Season', 'TeamID']
        ).rename(columns={'PriorExp': 'Team2TourneyExp'}).drop(columns='TeamID')
        
        # Create difference feature
        self.games['TourneyExpDiff'] = self.games['Team1TourneyExp'] - self.games['Team2TourneyExp']
        
        # Fill missing values
        self.games = self.games.fillna(0)


def add_momentum_features(self):
    """Add team momentum features (recent performance)"""
    print("Adding momentum features...")
    
    if self.season_results.empty:
        print("Warning: No season results available for momentum features")
        return
    
    # Sort by season and day
    season_games = self.season_results.copy()
    season_games = season_games.sort_values(['Season', 'DayNum'])
    
    # Calculate team's recent win rate (last N games)
    window_sizes = [3, 5, 10]
    for window in window_sizes:
        # Process winning teams
        win_momentum = []
        for season in season_games['Season'].unique():
            season_data = season_games[season_games['Season'] == season]
            for team in pd.concat([season_data['WTeamID'], season_data['LTeamID']]).unique():
                team_wins = season_data[season_data['WTeamID'] == team]
                team_losses = season_data[season_data['LTeamID'] == team]
                
                # Create team game log with results (1 for win, 0 for loss)
                team_games = pd.concat([
                    pd.DataFrame({'TeamID': team, 'DayNum': team_wins['DayNum'], 'Result': 1}),
                    pd.DataFrame({'TeamID': team, 'DayNum': team_losses['DayNum'], 'Result': 0})
                ]).sort_values('DayNum')
                
                # Calculate rolling win rate
                if len(team_games) > 0:
                    team_games[f'WinRate{window}'] = team_games['Result'].rolling(
                        window=min(window, len(team_games)), min_periods=1).mean()
                    
                    # Add to momentum data
                    for _, row in team_games.iterrows():
                        win_momentum.append({
                            'Season': season,
                            'TeamID': team,
                            'DayNum': row['DayNum'],
                            f'WinRate{window}': row[f'WinRate{window}']
                        })
        
        # Convert to DataFrame
        momentum_df = pd.DataFrame(win_momentum)
        
        # Add to games data
        if not self.games.empty:
            # Add team 1 momentum
            self.games = pd.merge(
                self.games,
                momentum_df,
                how='left',
                left_on=['Season', 'Team1', 'DayNum'],
                right_on=['Season', 'TeamID', 'DayNum']
            ).rename(columns={f'WinRate{window}': f'Team1WinRate{window}'}).drop(columns='TeamID')
            
            # Add team 2 momentum
            self.games = pd.merge(
                self.games,
                momentum_df,
                how='left',
                left_on=['Season', 'Team2', 'DayNum'],
                right_on=['Season', 'TeamID', 'DayNum']
            ).rename(columns={f'WinRate{window}': f'Team2WinRate{window}'}).drop(columns='TeamID')
            
            # Create difference feature
            self.games[f'WinRate{window}Diff'] = (
                self.games[f'Team1WinRate{window}'] - self.games[f'Team2WinRate{window}']
            )
    
    # Fill missing values
    if not self.games.empty:
        self.games = self.games.fillna(0)


def create_ensemble_prediction(self, models_list=None):
    """Create an ensemble prediction using multiple models"""
    print("Creating ensemble prediction...")
    
    if models_list is None or len(models_list) == 0:
        print("No models provided for ensemble")
        return
    
    if self.games.empty or not self.feature_cols:
        print("Error: No game data or features available for ensemble prediction")
        return
    
    # Prepare features
    X = self.games[self.feature_cols].fillna(0)
    X_imputed = self.imputer.fit_transform(X)
    X_scaled = self.scaler.fit_transform(X_imputed)
    y = self.games['Pred']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Train each model and collect predictions
    train_preds = np.zeros((len(y_train), len(models_list)))
    test_preds = np.zeros((len(y_test), len(models_list)))
    
    for i, model in enumerate(models_list):
        # Train model
        model.fit(X_train, y_train)
        
        # Generate predictions
        train_preds[:, i] = model.predict(X_train).clip(0.001, 0.999)
        test_preds[:, i] = model.predict(X_test).clip(0.001, 0.999)
    
    # Dynamic weighting of models based on prediction confidence
    def dynamic_weight(predictions, base_weights=None):
        if base_weights is None:
            base_weights = np.ones(predictions.shape[1]) / predictions.shape[1]
        
        # Calculate confidence (distance from 0.5)
        confidence = abs(predictions - 0.5)
        
        # Normalize confidence
        normalized_confidence = confidence / confidence.sum(axis=1, keepdims=True)
        
        # Adjust weights
        adjusted_weights = 0.7 * np.array(base_weights) + 0.3 * normalized_confidence
        
        # Normalize weights
        return adjusted_weights / adjusted_weights.sum(axis=1, keepdims=True)
    
    # Calculate base weights using test set performance
    base_weights = []
    for i in range(len(models_list)):
        score = brier_score_loss(y_test, test_preds[:, i])
        # Invert score since lower is better
        base_weights.append(1.0 / score)
    
    # Normalize base weights
    base_weights = np.array(base_weights) / sum(base_weights)
    
    # Apply dynamic weighting
    test_weights = dynamic_weight(test_preds, base_weights)
    
    # Generate weighted ensemble prediction
    ensemble_preds = np.sum(test_preds * test_weights, axis=1)
    
    # Evaluate ensemble
    ensemble_brier = brier_score_loss(y_test, ensemble_preds)
    print(f'Ensemble Brier Score: {ensemble_brier:.4f}')
    
    # Compare with individual models
    print("Individual model Brier scores:")
    for i, model in enumerate(models_list):
        individual_brier = brier_score_loss(y_test, test_preds[:, i])
        print(f'  Model {i+1}: {individual_brier:.4f}')
    
    # Set ensemble model as main model if it performs better
    if ensemble_brier < min([brier_score_loss(y_test, test_preds[:, i]) for i in range(len(models_list))]):
        print("Ensemble model outperforms individual models, using as final model")
        
        # Create a function for the ensemble prediction
        def ensemble_predict(X):
            individual_preds = np.zeros((X.shape[0], len(models_list)))
            for i, model in enumerate(models_list):
                individual_preds[:, i] = model.predict(X).clip(0.001, 0.999)
            
            weights = dynamic_weight(individual_preds, base_weights)
            return np.sum(individual_preds * weights, axis=1)
        
        # Store the function and models for later use
        self.ensemble_models = models_list
        self.base_weights = base_weights
        self.ensemble_predict = ensemble_predict


def _plot_feature_importance(self, top_n=20):
    """Plot feature importance"""
    if not hasattr(self.main_model, 'feature_importances_'):
        print("Model does not support feature importance visualization")
        return
    
    importances = self.main_model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': self.feature_cols,
        'importance': importances
    })
    
    feature_importance_df = feature_importance_df.sort_values('importance', ascending=False).head(top_n)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance_df, palette='viridis')
    plt.title(f'Top {top_n} Feature Importances')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show()

def _plot_calibration_curve(self, y_true, y_proba, n_bins=10):
    """Plot calibration curve"""
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

def _plot_prediction_distribution(self, predictions, title="Distribution of Predictions"):
    """Plot the distribution of model predictions"""
    plt.figure(figsize=(8, 6))
    sns.histplot(predictions, kde=True, color='skyblue')
    plt.title(title)
    plt.xlabel('Predicted Probability')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

def plot_roc_curve(self, y_true, y_proba, title="ROC Curve"):
    """Plot the Receiver Operating Characteristic (ROC) curve"""
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


def run_pipeline(self, optimize=False, use_ensemble=False, output_file='submission.csv'):
    """Run the complete prediction pipeline"""
    print("Running complete prediction pipeline...")
    
    # Load data
    self.load_data()
    
    # Prepare features
    self.prepare_features()
    
    # Add advanced features
    self.add_tournament_features()
    self.add_momentum_features()
    
    # Create models
    if use_ensemble:
        # Create multiple models for ensemble
        self.create_models('random_forest')
        rf_model = self.main_model
        
        self.create_models('catboost')
        cb_model = self.main_model
        
        # Create ensemble
        self.create_ensemble_prediction([rf_model, cb_model])
    else:
        # Create single model
        self.create_models('catboost')
        
        # Optimize hyperparameters if requested
        if optimize:
            self.hyperparameter_optimization(n_trials=10)
    
    # Train model
    self.train_model()
    
    # Generate predictions
    self.predict_submission(output_file)
    
    print("Pipeline complete!")
    
    print("Pipeline complete!")


def analyze_tournament_upsets(data_path):
    """Analyze historical tournament upsets to identify patterns"""
    # Load tournament results and seeds
    predictor = MarchMadnessPredictor(data_path)
    predictor.load_data()
    
    if predictor.tourney_results is None or predictor.seeds is None:
        print("Error: Tournament data not available")
        return
    
    # Get tournament games
    tourney_games = predictor.tourney_results.copy()
    
    # Add seed information
    tourney_games['WTeamSeed'] = tourney_games.apply(
        lambda r: predictor.seeds.get(f"{r['Season']}_{r['WTeamID']}", 16), axis=1)
    tourney_games['LTeamSeed'] = tourney_games.apply(
        lambda r: predictor.seeds.get(f"{r['Season']}_{r['LTeamID']}", 16), axis=1)
    
    # Identify upsets (when lower seed beats higher seed)
    tourney_games['IsUpset'] = tourney_games['WTeamSeed'] > tourney_games['LTeamSeed']
    
    # Calculate upset rate by seed matchup
    upset_by_matchup = tourney_games.groupby(['WTeamSeed', 'LTeamSeed']).agg(
        {'IsUpset': ['count', 'sum', 'mean']})
    upset_by_matchup.columns = ['TotalGames', 'Upsets', 'UpsetRate']
    upset_by_matchup = upset_by_matchup.reset_index()
    
    # Filter to matchups with enough games
    upset_by_matchup = upset_by_matchup[upset_by_matchup['TotalGames'] >= 5]
    
    # Sort by upset rate
    upset_by_matchup = upset_by_matchup.sort_values('UpsetRate', ascending=False)
    
    print("Top upset matchups by seed:")
    print(upset_by_matchup.head(10))
    
    # Visualization of upset rates
    plt.figure(figsize=(12, 6))
    sns.barplot(x='WTeamSeed', y='UpsetRate', hue='LTeamSeed', 
                data=upset_by_matchup[upset_by_matchup['TotalGames'] >= 10])
    plt.title('Upset Rates by Seed Matchup')
    plt.xlabel('Winning Team Seed')
    plt.ylabel('Upset Rate')
    plt.legend(title='Losing Team Seed')
    plt.tight_layout()
    plt.show()
    
    return upset_by_matchup


def analyze_feature_correlations(predictor):
    """Analyze correlations between features and prediction outcomes"""
    if predictor.games is None or predictor.feature_cols is None:
        print("Error: Game data or features not available")
        return
    
    # Get feature data
    features_df = predictor.games[predictor.feature_cols + ['Pred']]
    
    # Calculate correlations with prediction
    correlations = features_df.corr()['Pred'].sort_values(ascending=False)
    
    print("Top positive correlations with prediction outcome:")
    print(correlations.head(10))
    
    print("\nTop negative correlations with prediction outcome:")
    print(correlations.tail(10))
    
    # Visualize top correlations
    plt.figure(figsize=(12, 8))
    top_corr = pd.concat([correlations.head(10), correlations.tail(10)])
    sns.barplot(x=top_corr.values, y=top_corr.index)
    plt.title('Top Feature Correlations with Prediction Outcome')
    plt.xlabel('Correlation Coefficient')
    plt.tight_layout()
    plt.show()
    
    return correlations


def add_advanced_features(predictor):
    """Add additional advanced features to improve model performance"""
    # Add Elo ratings if not already present
    if predictor.games is not None and 'Team1Elo' not in predictor.games.columns:
        print("Adding Elo ratings...")
        
        # Initial Elo parameters
        k_factor = 20
        home_advantage = 100
        initial_elo = 1500
        elo_width = 400
        
        # Create a dictionary to store team Elos by season
        elos = {}
        
        # Process season data
        if predictor.season_results is not None:
            season_data = predictor.season_results.sort_values(['Season', 'DayNum'])
            
            for _, game in season_data.iterrows():
                season = game['Season']
                w_team = game['WTeamID']
                l_team = game['LTeamID']
                
                # Get team Elos, defaulting to initial Elo if not found
                if season not in elos:
                    elos[season] = {}
                
                w_elo = elos[season].get(w_team, initial_elo)
                l_elo = elos[season].get(l_team, initial_elo)
                
                # Calculate expected win probability
                expected_w = 1.0 / (1.0 + 10.0 ** ((l_elo - w_elo) / elo_width))
                
                # Update Elos
                new_w_elo = w_elo + k_factor * (1.0 - expected_w)
                new_l_elo = l_elo + k_factor * (0.0 - (1.0 - expected_w))
                
                # Store updated Elos
                elos[season][w_team] = new_w_elo
                elos[season][l_team] = new_l_elo
            
            # Add Elo features to games data
            if predictor.games is not None:
                # Create lists to store Elo values
                team1_elos = []
                team2_elos = []
                elo_diffs = []
                
                for _, game in predictor.games.iterrows():
                    season = game['Season']
                    team1 = game['Team1']
                    team2 = game['Team2']
                    
                    # Get Elos for the teams
                    team1_elo = elos.get(season, {}).get(team1, initial_elo)
                    team2_elo = elos.get(season, {}).get(team2, initial_elo)
                    
                    team1_elos.append(team1_elo)
                    team2_elos.append(team2_elo)
                    elo_diffs.append(team1_elo - team2_elo)
                
                # Add to DataFrame
                predictor.games['Team1Elo'] = team1_elos
                predictor.games['Team2Elo'] = team2_elos
                predictor.games['EloDiff'] = elo_diffs
                
                # Do the same for submission data
                if predictor.submission is not None:
                    team1_elos = []
                    team2_elos = []
                    elo_diffs = []
                    
                    for _, game in predictor.submission.iterrows():
                        season = game['Season']
                        team1 = game['Team1']
                        team2 = game['Team2']
                        
                        # Get Elos for the teams
                        team1_elo = elos.get(season, {}).get(team1, initial_elo)
                        team2_elo = elos.get(season, {}).get(team2, initial_elo)
                        
                        team1_elos.append(team1_elo)
                        team2_elos.append(team2_elo)
                        elo_diffs.append(team1_elo - team2_elo)
                    
                    # Add to DataFrame
                    predictor.submission['Team1Elo'] = team1_elos
                    predictor.submission['Team2Elo'] = team2_elos
                    predictor.submission['EloDiff'] = elo_diffs
    
    # Update feature columns
    if predictor.games is not None:
        exclude_cols = ['ID', 'DayNum', 'ST', 'Team1', 'Team2', 'IDTeams', 'IDTeam1', 'IDTeam2', 
                        'WTeamID', 'WScore', 'LTeamID', 'LScore', 'NumOT', 'Pred', 'ScoreDiff', 
                        'ScoreDiffNorm', 'WLoc']
        
        # Add any other columns that might be in the DataFrame but not needed as features
        if 'IDTeams_c_score' in predictor.games.columns:
            exclude_cols.append('IDTeams_c_score')
        
        # Get all columns except excluded ones
        predictor.feature_cols = [c for c in predictor.games.columns if c not in exclude_cols]
    
    print("Advanced features added successfully")
    return predictor

