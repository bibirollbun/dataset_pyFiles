# Import necessary libraries
import glob                          # For file pattern matching
import numpy as np                   # For numerical operations
import pandas as pd                  # For data manipulation and analysis

# Import scikit-learn modules for model building, evaluation, and preprocessing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss, mean_absolute_error, brier_score_loss
from sklearn.ensemble import RandomForestRegressor
from sklearn.isotonic import IsotonicRegression  # For probability calibration

import warnings
# Suppress warnings to keep the output clean during runtime
warnings.filterwarnings("ignore")

class TournamentPredictor:
    """
    Class to build, train, and generate predictions for the 2025 NCAA Basketball Tournaments.
    
    This class handles:
    - Loading and preprocessing the data from CSV files.
    - Creating useful features from the historical game data.
    - Training a Random Forest model to predict game outcomes.
    - Calibrating predicted probabilities using isotonic regression.
    - Generating submission files in the required format.
    """
    def __init__(self, data_path):
        """
        Initialize the TournamentPredictor with the data path and preprocessing/model objects.
        
        Parameters:
        - data_path: str, path pattern to the input CSV files (e.g., '/kaggle/input/march-machine-learning-mania-2025/**')
        """
        self.data_path = data_path  # File path to input data files
        self.data = None            # Dictionary to hold all loaded data files
        self.teams = None           # Combined team information (men's and women's)
        self.seeds = None           # Dictionary mapping season and team IDs to seed numbers
        self.games = None           # DataFrame to hold combined game results (both season and tournament)
        self.sub = None             # DataFrame for the submission file
        self.gb = None              # Aggregated game statistics by team pairing
        self.col = None             # List of feature column names used for training
        
        # Preprocessing objects:
        # SimpleImputer: to fill in missing values (using mean strategy)
        self.imputer = SimpleImputer(strategy='mean')
        # StandardScaler: to standardize features (mean=0, variance=1)
        self.scaler = StandardScaler()
        
        # Create a Random Forest Regressor model with specific hyperparameters:
        self.model = RandomForestRegressor(
            n_estimators=235,       # Number of trees in the forest
            random_state=42,        # Seed for reproducibility
            max_depth=15,           # Limit the depth of each tree to prevent overfitting
            min_samples_split=2,    # Minimum number of samples required to split an internal node
            max_features='auto'     # Use sqrt(n_features) for selecting features at each split
        )

    def load_data(self):
        """
        Load and preprocess all necessary data from CSV files.
        
        This method:
        - Loads multiple CSV files from the specified directory.
        - Combines data from both men's and women's teams.
        - Processes team names, seeds, and game results.
        - Creates various features and identifiers needed for model training and submission.
        """
        # Find all CSV files matching the data path pattern
        files = glob.glob(self.data_path)
        
        # Load each CSV file into a dictionary with keys based on the file name (without extension)
        self.data = {
            p.split('/')[-1].split('.')[0]: pd.read_csv(p, encoding='latin-1')
            for p in files
        }
        
        # Combine men's and women's team data into one DataFrame
        teams = pd.concat([self.data['MTeams'], self.data['WTeams']])
        # Combine team spellings for consistency
        teams_spelling = pd.concat([self.data['MTeamSpellings'], self.data['WTeamSpellings']])
        # Group by TeamID and count occurrences of team name spellings (could be used as a quality metric)
        teams_spelling = teams_spelling.groupby(by='TeamID', as_index=False)['TeamNameSpelling'].count()
        teams_spelling.columns = ['TeamID', 'TeamNameCount']
        # Merge the teams data with the spellings count
        self.teams = pd.merge(teams, teams_spelling, how='left', on=['TeamID'])
        # Delete temporary DataFrame to free memory
        del teams_spelling
        
        # Combine season and tournament results for both compact and detailed formats
        season_cresults = pd.concat([self.data['MRegularSeasonCompactResults'], self.data['WRegularSeasonCompactResults']])
        season_dresults = pd.concat([self.data['MRegularSeasonDetailedResults'], self.data['WRegularSeasonDetailedResults']])
        tourney_cresults = pd.concat([self.data['MNCAATourneyCompactResults'], self.data['WNCAATourneyCompactResults']])
        tourney_dresults = pd.concat([self.data['MNCAATourneyDetailedResults'], self.data['WNCAATourneyDetailedResults']])
        
        # Load seeds data from both men's and women's tournaments
        seeds_df = pd.concat([self.data['MNCAATourneySeeds'], self.data['WNCAATourneySeeds']])
        # Also load game cities and seasons for potential future use
        gcities = pd.concat([self.data['MGameCities'], self.data['WGameCities']])
        seasons = pd.concat([self.data['MSeasons'], self.data['WSeasons']])
        
        # Create a dictionary for seeds with keys formatted as "Season_TeamID" and value as the seed number.
        # The seed string in the CSV (e.g., "W01") has its first character removed to extract the numeric seed.
        self.seeds = {
            '_'.join(map(str, [int(k1), k2])): int(v[1:3])
            for k1, v, k2 in seeds_df[['Season', 'Seed', 'TeamID']].values
        }
        
        # Load additional data: cities and the sample submission file.
        # The sample submission provides the template for the final predictions.
        cities = self.data['Cities']
        self.sub = self.data['SampleSubmissionStage1']
        # Free up memory by deleting unused variables
        del seeds_df, cities
        
        # Mark the type of results: 'S' for season games and 'T' for tournament games
        season_cresults['ST'] = 'S'
        season_dresults['ST'] = 'S'
        tourney_cresults['ST'] = 'T'
        tourney_dresults['ST'] = 'T'
        
        # We are using the detailed results for feature processing
        self.games = pd.concat((season_dresults, tourney_dresults), axis=0, ignore_index=True)
        # Reset index for consistency
        self.games.reset_index(drop=True, inplace=True)
        # Map the location column 'WLoc' to numerical values (A=1, H=2, N=3)
        self.games['WLoc'] = self.games['WLoc'].map({'A': 1, 'H': 2, 'N': 3})
        
        # Create unique IDs for each game and extract team identifiers
        # ID is created by concatenating Season and the sorted team IDs
        self.games['ID'] = self.games.apply(
            lambda r: '_'.join(map(str, [r['Season']] + sorted([r['WTeamID'], r['LTeamID']]))), axis=1
        )
        # Create an ID based solely on team IDs (regardless of season)
        self.games['IDTeams'] = self.games.apply(
            lambda r: '_'.join(map(str, sorted([r['WTeamID'], r['LTeamID']]))), axis=1
        )
        # Define Team1 as the team with the lower team ID and Team2 as the other team
        self.games['Team1'] = self.games.apply(
            lambda r: sorted([r['WTeamID'], r['LTeamID']])[0], axis=1
        )
        self.games['Team2'] = self.games.apply(
            lambda r: sorted([r['WTeamID'], r['LTeamID']])[1], axis=1
        )
        # Create IDs that include season information for each team (for mapping seeds)
        self.games['IDTeam1'] = self.games.apply(
            lambda r: '_'.join(map(str, [r['Season'], r['Team1']])), axis=1
        )
        self.games['IDTeam2'] = self.games.apply(
            lambda r: '_'.join(map(str, [r['Season'], r['Team2']])), axis=1
        )
        # Map seeds to teams using the precomputed dictionary; if missing, fill with 0
        self.games['Team1Seed'] = self.games['IDTeam1'].map(self.seeds).fillna(0)
        self.games['Team2Seed'] = self.games['IDTeam2'].map(self.seeds).fillna(0)
        
        # Create additional game-level features:
        # - Score difference between winning and losing teams
        self.games['ScoreDiff'] = self.games['WScore'] - self.games['LScore']
        # - Binary indicator for whether the team with the lower TeamID (Team1) won the game
        self.games['Pred'] = self.games.apply(
            lambda r: 1.0 if sorted([r['WTeamID'], r['LTeamID']])[0] == r['WTeamID'] else 0.0, axis=1
        )
        # Normalize score difference so that it is positive when Team1 wins
        self.games['ScoreDiffNorm'] = self.games.apply(
            lambda r: r['ScoreDiff'] * -1 if r['Pred'] == 0.0 else r['ScoreDiff'], axis=1
        )
        # Compute the difference in seeds between Team1 and Team2
        self.games['SeedDiff'] = self.games['Team1Seed'] - self.games['Team2Seed']
        # Replace any remaining missing values with -1
        self.games = self.games.fillna(-1)
        
        # ------------------------
        # Aggregate game statistics for each team pairing.
        # For a list of columns capturing various game stats:
        c_score_col = [
            'NumOT', 'WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WOR', 'WDR', 'WAst', 
            'WTO', 'WStl', 'WBlk', 'WPF', 'LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA', 
            'LOR', 'LDR', 'LAst', 'LTO', 'LStl', 'LBlk', 'LPF'
        ]
        # Define aggregation functions to be applied on each of these columns
        c_score_agg = ['sum', 'mean', 'median', 'max', 'min', 'std', 'skew', 'nunique']
        # Group by team pairing (IDTeams) and aggregate the statistics
        self.gb = self.games.groupby(by=['IDTeams']).agg({k: c_score_agg for k in c_score_col}).reset_index()
        # Rename aggregated columns to have a suffix indicating they belong to aggregated game statistics
        self.gb.columns = [''.join(c) + '_c_score' for c in self.gb.columns]
        
        # Filter the games DataFrame to keep only tournament games (where ST is 'T')
        self.games = self.games[self.games['ST'] == 'T']
        
        # ------------------------
        # Process the submission file which provides the template for predictions:
        # - Set a default location value (3, representing 'Neutral')
        self.sub['WLoc'] = 3
        # - Extract the season from the submission ID (first part before the underscore)
        self.sub['Season'] = self.sub['ID'].map(lambda x: x.split('_')[0]).astype(int)
        # - Extract Team1 and Team2 from the submission ID (second and third parts)
        self.sub['Team1'] = self.sub['ID'].map(lambda x: x.split('_')[1])
        self.sub['Team2'] = self.sub['ID'].map(lambda x: x.split('_')[2])
        # Create a combined ID for teams similar to the games DataFrame
        self.sub['IDTeams'] = self.sub.apply(
            lambda r: '_'.join(map(str, [r['Team1'], r['Team2']])), axis=1
        )
        # Create season-specific IDs for each team for seed mapping
        self.sub['IDTeam1'] = self.sub.apply(
            lambda r: '_'.join(map(str, [r['Season'], r['Team1']])), axis=1
        )
        self.sub['IDTeam2'] = self.sub.apply(
            lambda r: '_'.join(map(str, [r['Season'], r['Team2']])), axis=1
        )
        # Map seed values to teams; if not found, use 0 as default
        self.sub['Team1Seed'] = self.sub['IDTeam1'].map(self.seeds).fillna(0)
        self.sub['Team2Seed'] = self.sub['IDTeam2'].map(self.seeds).fillna(0)
        # Calculate seed difference for the matchup
        self.sub['SeedDiff'] = self.sub['Team1Seed'] - self.sub['Team2Seed']
        # Replace any missing values in the submission DataFrame with -1
        self.sub = self.sub.fillna(-1)
        
        # Merge the aggregated game statistics into both the games and submission DataFrames.
        # This step adds historical matchup statistics as features for prediction.
        self.games = pd.merge(self.games, self.gb, how='left', left_on='IDTeams', right_on='IDTeams_c_score')
        self.sub = pd.merge(self.sub, self.gb, how='left', left_on='IDTeams', right_on='IDTeams_c_score')
        
        # Define the list of feature columns to use for training.
        # Exclude columns that are identifiers or raw scores which are not used as features.
        exclude_cols = [
            'ID', 'DayNum', 'ST', 'Team1', 'Team2', 'IDTeams', 'IDTeam1', 'IDTeam2',
            'WTeamID', 'WScore', 'LTeamID', 'LScore', 'NumOT', 'Pred', 'ScoreDiff', 
            'ScoreDiffNorm', 'WLoc'
        ] + c_score_col
        self.col = [c for c in self.games.columns if c not in exclude_cols]
        
        print("Data loading and preprocessing completed.")

    def train_model(self):
        """
        Train the Random Forest model using the tournament games data.
        
        This method:
        - Prepares the training features and target variable.
        - Applies imputation and scaling.
        - Fits the Random Forest model.
        - Calibrates the predicted probabilities using Isotonic Regression.
        - Prints evaluation metrics such as Log Loss, MAE, Brier Score, and cross-validated MSE.
        """
        # Select the training features from the games DataFrame and fill missing values with -1
        X = self.games[self.col].fillna(-1)
        # Apply imputation (fill missing values) based on the training data's mean
        X_imputed = self.imputer.fit_transform(X)
        # Scale the features to standardize them (mean=0, variance=1)
        X_scaled = self.scaler.fit_transform(X_imputed)
        # Target variable: whether the team with the lower TeamID (Team1) won the game
        y = self.games['Pred']
        
        # Fit the Random Forest model on the preprocessed features
        self.model.fit(X_scaled, y)
        # Predict on the training set and clip predictions to avoid extreme probabilities
        pred = self.model.predict(X_scaled).clip(0.001, 0.999)
        
        # ---- Calibration Step ----
        # Use Isotonic Regression to calibrate the predicted probabilities.
        # Note: Ideally, calibration should be performed on a separate holdout set.
        ir = IsotonicRegression(out_of_bounds='clip')
        ir.fit(pred, y)  # Fit the calibration model using predictions and true labels
        pred_cal = ir.transform(pred)  # Transform the predictions
        
        # Calculate and print various evaluation metrics:
        print(f'Log Loss: {log_loss(y, pred_cal):.4f}')
        print(f'Mean Absolute Error: {mean_absolute_error(y, pred_cal):.4f}')
        print(f'Brier Score: {brier_score_loss(y, pred_cal):.4f}')
        # Use cross-validation to evaluate model performance (MSE metric)
        cv_scores = cross_val_score(self.model, X_scaled, y, cv=5, scoring='neg_mean_squared_error')
        print(f'Cross-validated MSE: {-cv_scores.mean():.4f}')

    def predict_submission(self, output_file='submission.csv'):
        """
        Generate predictions for the submission file and save the results to a CSV file.
        
        This method:
        - Prepares the submission features.
        - Applies the same imputation and scaling as used in training.
        - Generates predictions and calibrates them using Isotonic Regression.
        - Writes the final predictions to the specified CSV file.
        
        Parameters:
        - output_file: str, file name for the output submission CSV (default: 'submission.csv')
        """
        # Prepare submission features and fill missing values
        sub_X = self.sub[self.col].fillna(-1)
        # Apply the imputer (using the training data statistics)
        sub_X_imputed = self.imputer.transform(sub_X)
        # Scale the submission features using the previously fitted scaler
        sub_X_scaled = self.scaler.transform(sub_X_imputed)
        # Predict probabilities using the trained model and clip the values to a reasonable range
        preds = self.model.predict(sub_X_scaled).clip(0.01, 0.99)
        
        # ---- Calibration for Submission Predictions ----
        # Refit the isotonic regression on the training data predictions for consistency.
        ir = IsotonicRegression(out_of_bounds='clip')
        # Prepare training features again for calibration purposes
        X_train = self.imputer.fit_transform(self.games[self.col].fillna(-1))
        X_train_scaled = self.scaler.fit_transform(X_train)
        # Get predictions on the training set
        train_preds = self.model.predict(X_train_scaled).clip(0.001, 0.999)
        # Fit the isotonic regression model using training predictions and true labels
        ir.fit(train_preds, self.games['Pred'])
        # Calibrate the submission predictions
        preds_cal = ir.transform(preds)
        
        # Assign calibrated predictions to the submission DataFrame
        self.sub['Pred'] = preds_cal
        # Save the submission file with only the required columns (ID and Pred)
        self.sub[['ID', 'Pred']].to_csv(output_file, index=False)
        print(f"Submission file saved to {output_file}")

    def run_all(self):
        """
        Run the complete pipeline:
        1. Load and preprocess data.
        2. Train the model.
        3. Generate submission predictions and save them to a file.
        """
        self.load_data()
        self.train_model()
        self.predict_submission()

# ------------------------
# Main block: Execute the full pipeline if this script is run directly
if __name__ == "__main__":
    # Specify the data path (adjust the path as needed for your environment)
    data_path = '/kaggle/input/march-machine-learning-mania-2025/**'
    # Initialize the predictor with the data path
    predictor = TournamentPredictor(data_path)
    # Run the full pipeline: data loading, training, and prediction generation
    predictor.run_all()

