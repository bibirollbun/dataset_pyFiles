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
import numpy as np
import pandas as pd
from scipy.sparse import issparse
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
import joblib

# Set up paths
DATA_DIR = "/kaggle/input/march-machine-learning-mania-2025/"
OUTPUT_DIR = "/kaggle/working/"

# Configuration
RANDOM_STATE = 42
N_JOBS = -1
TEST_YEAR = 2025

class NCAAPredictor:
    def __init__(self):
        self.models = {}
        self.feature_processor = None
        self.cities_df = None
        self.teams_df = None
        self.seeds_df = None
        self.rankings_df = None
        self.team_games = None
        
    def load_data(self):
        """Load all required datasets"""
        print("Loading data...")
        
        # Core datasets
        self.cities_df = pd.read_csv(os.path.join(DATA_DIR, "Cities.csv")).copy()
        m_teams = pd.read_csv(os.path.join(DATA_DIR, "MTeams.csv")).copy()
        w_teams = pd.read_csv(os.path.join(DATA_DIR, "WTeams.csv")).copy()
        self.teams_df = pd.concat([m_teams.assign(Gender='M'), 
                                 w_teams.assign(Gender='W')]).copy()
        
        # Game results
        game_dfs = {
            'M': {
                'reg': pd.read_csv(os.path.join(DATA_DIR, "MRegularSeasonDetailedResults.csv")).copy(),
                'tourney': pd.read_csv(os.path.join(DATA_DIR, "MNCAATourneyDetailedResults.csv")).copy(),
                'secondary': pd.read_csv(os.path.join(DATA_DIR, "MSecondaryTourneyCompactResults.csv")).copy()
            },
            'W': {
                'reg': pd.read_csv(os.path.join(DATA_DIR, "WRegularSeasonDetailedResults.csv")).copy(),
                'tourney': pd.read_csv(os.path.join(DATA_DIR, "WNCAATourneyDetailedResults.csv")).copy(),
                'secondary': pd.read_csv(os.path.join(DATA_DIR, "WSecondaryTourneyCompactResults.csv")).copy()
            }
        }
        
        # Tournament structure
        m_seeds = pd.read_csv(os.path.join(DATA_DIR, "MNCAATourneySeeds.csv")).copy()
        w_seeds = pd.read_csv(os.path.join(DATA_DIR, "WNCAATourneySeeds.csv")).copy()
        self.seeds_df = pd.concat([m_seeds.assign(Gender='M'), 
                                  w_seeds.assign(Gender='W')]).copy()
        
        # Rankings
        self.rankings_df = pd.read_csv(os.path.join(DATA_DIR, "MMasseyOrdinals.csv")).copy()
        
        # Geographic data
        m_game_cities = pd.read_csv(os.path.join(DATA_DIR, "MGameCities.csv")).copy()
        w_game_cities = pd.read_csv(os.path.join(DATA_DIR, "WGameCities.csv")).copy()
        game_cities_df = pd.concat([m_game_cities.assign(Gender='M'), 
                                   w_game_cities.assign(Gender='W')]).copy()
        
        return game_dfs, game_cities_df
    
    def preprocess_data(self, game_dfs, game_cities_df):
        """Clean and merge datasets"""
        print("Preprocessing data...")
        
        # Process game data
        all_games = []
        for gender in ['M', 'W']:
            for game_type in ['reg', 'tourney', 'secondary']:
                df = game_dfs[gender][game_type].copy()
                df['GameType'] = game_type
                df['Gender'] = gender
                all_games.append(df)
        
        games_df = pd.concat(all_games).copy()
        
        # Add city information
        games_df = games_df.merge(
            game_cities_df,
            on=['Season', 'DayNum', 'WTeamID', 'LTeamID', 'Gender'],
            how='left'
        ).copy()
        
        # Add team names
        team_names = self.teams_df[['TeamID', 'TeamName']].copy()
        games_df = games_df.merge(
            team_names.rename(columns={'TeamName': 'WTeamName'}),
            left_on='WTeamID',
            right_on='TeamID',
            how='left'
        ).drop(columns=['TeamID']).copy()
        
        games_df = games_df.merge(
            team_names.rename(columns={'TeamName': 'LTeamName'}),
            left_on='LTeamID',
            right_on='TeamID',
            how='left'
        ).drop(columns=['TeamID']).copy()
        
        return games_df
    
    def calculate_advanced_metrics(self, games_df):
        """Calculate advanced basketball statistics"""
        print("Calculating advanced metrics...")
        
        games_df = games_df.copy()
        
        # Calculate Four Factors for both teams
        games_df['WPoss'] = 0.96 * (games_df['WFGA'] + games_df['WTO'] + 0.44 * games_df['WFTA'] - games_df['WOR'])
        games_df['LPoss'] = 0.96 * (games_df['LFGA'] + games_df['LTO'] + 0.44 * games_df['LFTA'] - games_df['LOR'])
        
        # Offensive/Defensive Ratings
        games_df['WORTG'] = games_df['WScore'] / games_df['WPoss'] * 100
        games_df['WDRTG'] = games_df['LScore'] / games_df['WPoss'] * 100
        games_df['LORTG'] = games_df['LScore'] / games_df['LPoss'] * 100
        games_df['LDRTG'] = games_df['WScore'] / games_df['LPoss'] * 100
        
        # Effective Field Goal Percentage
        games_df['WeFG%'] = (games_df['WFGM'] + 0.5 * games_df['WFGM3']) / games_df['WFGA']
        games_df['LeFG%'] = (games_df['LFGM'] + 0.5 * games_df['LFGM3']) / games_df['LFGA']
        
        return games_df
    
    def create_team_features(self, games_df):
        """Create team-level features from game data"""
        print("Creating team features...")
        
        # Prepare both winner and loser perspectives
        winner_stats = games_df.copy()
        loser_stats = games_df.copy()
        
        # Rename columns for consistent team perspective
        rename_dict = {
            'WTeamID': 'TeamID',
            'LTeamID': 'OppTeamID',
            'WScore': 'PointsFor',
            'LScore': 'PointsAgainst',
            'WORTG': 'ORTG',
            'WDRTG': 'DRTG',
            'WeFG%': 'eFG%',
            'LORTG': 'OppORTG',
            'LDRTG': 'OppDRTG',
            'LeFG%': 'OppeFG%',
            'Win': 1
        }
        
        winner_stats = winner_stats.rename(columns=rename_dict)
        winner_stats['Win'] = 1
        
        reverse_rename = {v: k for k, v in rename_dict.items() if k.startswith('W')}
        reverse_rename.update({
            'LTeamID': 'TeamID',
            'WTeamID': 'OppTeamID',
            'PointsFor': 'LScore',
            'PointsAgainst': 'WScore',
            'Win': 0
        })
        
        loser_stats = loser_stats.rename(columns=reverse_rename)
        loser_stats['Win'] = 0
        
        # Combine into single team-game dataframe
        team_games = pd.concat([winner_stats, loser_stats]).copy()
        team_games = team_games.sort_values(['TeamID', 'Season', 'DayNum'])
        
        # Define features to calculate rolling stats for
        stat_cols = ['ORTG', 'DRTG', 'eFG%', 'OppORTG', 'OppDRTG', 'OppeFG%']
        
        # Calculate rolling averages (last 10 games)
        for col in stat_cols:
            team_games[f'Roll10_{col}'] = (
                team_games.groupby('TeamID')[col]
                .transform(lambda x: x.rolling(10, min_periods=3).mean()))
        
        # Calculate season averages
        season_stats = (
            team_games.groupby(['TeamID', 'Season'])
            .agg({
                'ORTG': 'mean',
                'DRTG': 'mean',
                'eFG%': 'mean',
                'Win': 'mean',
                'PointsFor': 'mean',
                'PointsAgainst': 'mean'
            })
            .add_prefix('Season_')
            .reset_index()
        )
        
        # Merge back with game data
        team_games = team_games.merge(
            season_stats,
            on=['TeamID', 'Season'],
            how='left'
        ).copy()
        
        return team_games
    
    def create_matchup_features(self, team_games):
        """Create features for each team matchup"""
        print("Creating matchup features...")
        
        # Get all unique matchups
        matchups = team_games[['Season', 'DayNum', 'TeamID', 'OppTeamID', 'Win', 'Gender']].copy()
        
        # Merge with team features
        for side in ['Team', 'OppTeam']:
            cols = [c for c in team_games.columns if c.startswith('Roll10_') or c.startswith('Season_')]
            cols = ['TeamID', 'Season', 'DayNum'] + cols
            
            temp_df = team_games[cols].copy()
            temp_df.columns = [f"{side}_{c}" if c not in ['Season', 'DayNum'] else c for c in temp_df.columns]
            
            matchups = matchups.merge(
                temp_df,
                left_on=['Season', 'DayNum', f'{side}ID'],
                right_on=['Season', 'DayNum', f'{side}_TeamID'],
                how='left'
            ).copy()
        
        # Calculate differential features
        for stat in ['ORTG', 'DRTG', 'eFG%']:
            matchups[f'{stat}_Diff'] = matchups[f'Team_Season_{stat}'] - matchups[f'OppTeam_Season_{stat}']
            matchups[f'Roll10_{stat}_Diff'] = matchups[f'Team_Roll10_{stat}'] - matchups[f'OppTeam_Roll10_{stat}']
        
        # Add seed information if available
        matchups = matchups.merge(
            self.seeds_df.rename(columns={'TeamID': 'TeamID', 'Seed': 'TeamSeed'}),
            on=['Season', 'TeamID', 'Gender'],
            how='left'
        ).copy()
        
        matchups = matchups.merge(
            self.seeds_df.rename(columns={'TeamID': 'OppTeamID', 'Seed': 'OppSeed'}),
            on=['Season', 'OppTeamID', 'Gender'],
            how='left'
        ).copy()
        
        # Calculate seed differential
        def extract_seed_number(seed_str):
            if pd.isna(seed_str):
                return np.nan
            return int(''.join(filter(str.isdigit, str(seed_str))))
        
        matchups['TeamSeedNum'] = matchups['TeamSeed'].apply(extract_seed_number)
        matchups['OppSeedNum'] = matchups['OppSeed'].apply(extract_seed_number)
        matchups['SeedDiff'] = matchups['TeamSeedNum'] - matchups['OppSeedNum']
        
        return matchups
    
    def build_feature_pipeline(self):
        """Build preprocessing pipeline for features"""
        print("Building feature pipeline...")
        
        # Numeric features
        numeric_features = [
            'Team_Season_ORTG', 'Team_Season_DRTG', 'Team_Season_eFG%',
            'OppTeam_Season_ORTG', 'OppTeam_Season_DRTG', 'OppTeam_Season_eFG%',
            'ORTG_Diff', 'DRTG_Diff', 'eFG%_Diff', 'SeedDiff'
        ]
        
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Categorical features
        categorical_features = ['TeamSeed', 'OppSeed']
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))  # Force dense output
        ])
        
        # Combine preprocessing steps
        self.feature_processor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            sparse_threshold=0  # Force dense output
        )
        
        return self.feature_processor
    
    def train_models(self, X, y):
        """Train ensemble of models"""
        print("Training models...")
        
        # Convert sparse matrix to dense if needed
        if issparse(X):
            X = X.toarray()
        
        # Ensure we're working with writable arrays
        #if not X.flags.writeable:
            #X = np.array(X, copy=True)
        #if not y.flags.writeable:
            #y = np.array(y, copy=True)
        
        # Define base models
        base_models = [
            ('xgb', XGBClassifier(
                n_estimators=1000,
                learning_rate=0.01,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=RANDOM_STATE,
                eval_metric='logloss',
                n_jobs=N_JOBS
            )),
            ('lgbm', LGBMClassifier(
                n_estimators=1000,
                learning_rate=0.01,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=RANDOM_STATE,
                n_jobs=N_JOBS
            ))
        ]
        
        # Define meta model
        meta_model = LogisticRegression(penalty='l2', C=1.0, random_state=RANDOM_STATE)
        
        # Create and fit stacking ensemble directly without cross_val_predict
        self.models['ensemble'] = StackingClassifier(
            estimators=base_models,
            final_estimator=meta_model,
            #cv=TimeSeriesSplit(n_splits=3),
            n_jobs=N_JOBS
        )
        
        # Fit the model
        self.models['ensemble'].fit(X, y)
        
        return self.models
    
    def generate_predictions(self, test_year):
        """Generate predictions for all possible matchups"""
        print(f"Generating predictions for {test_year}...")
        
        # Get all teams
        teams = self.teams_df['TeamID'].unique()
        
        # Create all possible matchups
        matchups = []
        for i, team1 in enumerate(teams):
            for team2 in teams[i+1:]:
                if team1 < team2:
                    matchups.append((test_year, team1, team2))
        
        # Create feature matrix for predictions
        X_pred = pd.DataFrame(matchups, columns=['Season', 'TeamID', 'OppTeamID']).copy()
        
        # Add team features
        latest_season = self.team_games['Season'].max()
        team_features = (
            self.team_games[self.team_games['Season'] == latest_season]
            .drop_duplicates(subset=['TeamID'])
            .copy()
        )
        
        for side in ['Team', 'OppTeam']:
            cols = [c for c in team_features.columns if c.startswith('Season_')]
            cols = ['TeamID'] + cols
            
            temp_df = team_features[cols].copy()
            temp_df.columns = [f"{side}_{c}" if c != 'TeamID' else f"{side}ID" for c in temp_df.columns]
            
            X_pred = X_pred.merge(
                temp_df,
                on=f"{side}ID",
                how='left'
            ).copy()
        
        # Add seed information
        seeds = self.seeds_df[self.seeds_df['Season'] == test_year].copy()
        
        X_pred = X_pred.merge(
            seeds.rename(columns={'TeamID': 'TeamID', 'Seed': 'TeamSeed'}),
            on=['Season', 'TeamID'],
            how='left'
        ).copy()
        
        X_pred = X_pred.merge(
            seeds.rename(columns={'TeamID': 'OppTeamID', 'Seed': 'OppSeed'}),
            on=['Season', 'OppTeamID'],
            how='left'
        ).copy()
        
        # Calculate differential features
        for stat in ['ORTG', 'DRTG', 'eFG%']:
            X_pred[f'{stat}_Diff'] = X_pred[f'Team_Season_{stat}'] - X_pred[f'OppTeam_Season_{stat}']
        
        # Process seed differential
        def extract_seed_num(seed):
            if pd.isna(seed):
                return np.nan
            return int(''.join(filter(str.isdigit, str(seed))))
        
        X_pred['TeamSeedNum'] = X_pred['TeamSeed'].apply(extract_seed_num)
        X_pred['OppSeedNum'] = X_pred['OppSeed'].apply(extract_seed_num)
        X_pred['SeedDiff'] = X_pred['TeamSeedNum'] - X_pred['OppSeedNum']
        
        # Preprocess features
        X_processed = self.feature_processor.transform(X_pred)
        
        # Convert sparse matrix to dense if needed
        if issparse(X_processed):
            X_processed = X_processed.toarray()
        
        # Make predictions
        preds = self.models['ensemble'].predict_proba(X_processed)[:, 1]
        
        # Create submission
        submission = pd.DataFrame({
            'ID': [f"{test_year}_{min(t1,t2)}_{max(t1,t2)}" for _, t1, t2 in matchups],
            'Pred': preds
        }).copy()
        
        return submission
    
    def run_pipeline(self):
        """Execute full prediction pipeline"""
        try:
            # Load and preprocess data
            game_dfs, game_cities_df = self.load_data()
            games_df = self.preprocess_data(game_dfs, game_cities_df)
            
            # Calculate advanced metrics
            games_df = self.calculate_advanced_metrics(games_df)
            
            # Create team features
            self.team_games = self.create_team_features(games_df)
            
            # Create matchup features
            matchups = self.create_matchup_features(self.team_games)
            
            # Prepare training data
            X = matchups.drop(columns=['Win', 'Season', 'DayNum', 'TeamID', 'OppTeamID', 'Gender'])
            y = matchups['Win']
            
            # Build and fit feature processor
            self.build_feature_pipeline()
            X_processed = self.feature_processor.fit_transform(X)
            
            # Train models
            self.train_models(X_processed, y)
            
            # Generate predictions
            submission = self.generate_predictions(TEST_YEAR)
            
            # Save submission
            submission.to_csv(os.path.join(OUTPUT_DIR, "submission.csv"), index=False)
            print("Submission file saved successfully!")
            
            return submission
            
        except Exception as e:
            print(f"Error encountered: {str(e)}")
            raise

# Execute the pipeline
if __name__ == "__main__":
    try:
        predictor = NCAAPredictor()
        submission = predictor.run_pipeline()
        print(submission.head())
    except Exception as e:
        print(f"Pipeline failed: {str(e)}")




