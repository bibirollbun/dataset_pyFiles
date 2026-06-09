import os, re, pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import logging
from sklearn.metrics import mean_squared_error
from tqdm import tqdm  # progress bar

# RAPIDS cuDF for GPU preprocessing
import cudf

# Import CatBoost (GPU-enabled)
from catboost import CatBoostClassifier, Pool

# Clear existing logging handlers (if any)
if logging.getLogger().hasHandlers():
    logging.getLogger().handlers.clear()

# Configure logging to output to "ml_model_metrics.txt"
logging.basicConfig(
    filename='ml_model_metrics.txt',
    filemode='w',  # overwrite the file each time
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logging.info("Logging is now configured and should be stored in ml_model_metrics.txt")

class CatBoostPipeline:
    def __init__(self, data_path, hyperparams=None):
        """
        Initialize the pipeline.
        
        Parameters:
          data_path (str): Path to the data directory.
          hyperparams (dict): Dictionary of CatBoost hyperparameters.
        """
        self.data_path = data_path
        if hyperparams is None:
            self.hyperparams = {
                "iterations": 500,
                "learning_rate": 0.05,
                "depth": 6,
                "task_type": "GPU",
                "devices": '0',
                "verbose": 100
            }
        else:
            self.hyperparams = hyperparams
        
        # Feature definitions
        self.features_numeric = ['SeedA', 'SeedB', 'WinRatioA', 'GapAvgA', 
                                 'WinRatioB', 'GapAvgB', 'SeedDiff', 'WinRatioDiff', 'GapAvgDiff']
        self.cat_cols = ['TeamIdA', 'TeamIdB']
        self.features_gpu = self.features_numeric + [col + '_target' for col in self.cat_cols]
    
    def set_hyperparameters(self, new_params):
        """Update hyperparameters."""
        self.hyperparams.update(new_params)
    
    def _treat_seed(self, seed):
        return int(re.sub("[^0-9]", "", str(seed)))
    
    def load_data(self):
        """Read seeds, regular season and tournament results; compute season-level features; merge seeds; create mirrored matches; compute difference features."""
        # Read tournament seeds
        df_seeds = pd.concat([
            pd.read_csv(os.path.join(self.data_path, "MNCAATourneySeeds.csv")),
            pd.read_csv(os.path.join(self.data_path, "WNCAATourneySeeds.csv"))
        ], ignore_index=True)
        
        # Read regular season results
        df_season_results = pd.concat([
            pd.read_csv(os.path.join(self.data_path, "MRegularSeasonCompactResults.csv")),
            pd.read_csv(os.path.join(self.data_path, "WRegularSeasonCompactResults.csv"))
        ], ignore_index=True)
        df_season_results.drop(['NumOT', 'WLoc'], axis=1, inplace=True)
        df_season_results['ScoreGap'] = df_season_results['WScore'] - df_season_results['LScore']
        
        # Compute season-level features
        num_win = (df_season_results.groupby(['Season', 'WTeamID'])
                   .count().reset_index()[['Season', 'WTeamID', 'DayNum']]
                   .rename(columns={"DayNum": "NumWins", "WTeamID": "TeamID"}))
        num_loss = (df_season_results.groupby(['Season', 'LTeamID'])
                    .count().reset_index()[['Season', 'LTeamID', 'DayNum']]
                    .rename(columns={"DayNum": "NumLosses", "LTeamID": "TeamID"}))
        gap_win = (df_season_results.groupby(['Season', 'WTeamID'])
                   .mean().reset_index()[['Season', 'WTeamID', 'ScoreGap']]
                   .rename(columns={"ScoreGap": "GapWins", "WTeamID": "TeamID"}))
        gap_loss = (df_season_results.groupby(['Season', 'LTeamID'])
                    .mean().reset_index()[['Season', 'LTeamID', 'ScoreGap']]
                    .rename(columns={"ScoreGap": "GapLosses", "LTeamID": "TeamID"}))
        df_features_season = pd.concat([
            df_season_results.groupby(['Season', 'WTeamID']).count().reset_index()[['Season', 'WTeamID']].rename(columns={"WTeamID": "TeamID"}),
            df_season_results.groupby(['Season', 'LTeamID']).count().reset_index()[['Season', 'LTeamID']].rename(columns={"LTeamID": "TeamID"})
        ], axis=0).drop_duplicates().sort_values(['Season', 'TeamID']).reset_index(drop=True)
        df_features_season = df_features_season.merge(num_win, on=['Season', 'TeamID'], how='left')
        df_features_season = df_features_season.merge(num_loss, on=['Season', 'TeamID'], how='left')
        df_features_season = df_features_season.merge(gap_win, on=['Season', 'TeamID'], how='left')
        df_features_season = df_features_season.merge(gap_loss, on=['Season', 'TeamID'], how='left')
        df_features_season.fillna(0, inplace=True)
        df_features_season['WinRatio'] = df_features_season['NumWins'] / (df_features_season['NumWins'] + df_features_season['NumLosses'])
        df_features_season['GapAvg'] = ((df_features_season['NumWins'] * df_features_season['GapWins'] - 
                                         df_features_season['NumLosses'] * df_features_season['GapLosses']) /
                                        (df_features_season['NumWins'] + df_features_season['NumLosses']))
        df_features_season.drop(['NumWins', 'NumLosses', 'GapWins', 'GapLosses'], axis=1, inplace=True)
        self.df_features_season = df_features_season
        
        # Read tournament results
        df_tourney_results = pd.concat([
            pd.read_csv(os.path.join(self.data_path, "WNCAATourneyCompactResults.csv")),
            pd.read_csv(os.path.join(self.data_path, "MNCAATourneyCompactResults.csv"))
        ], ignore_index=True)
        df_tourney_results.drop(['NumOT', 'WLoc'], axis=1, inplace=True)
        df = df_tourney_results.copy()
        df = df[df['Season'] >= 2016].reset_index(drop=True)
        
        # Merge seeds
        df = pd.merge(df, df_seeds, how='left', left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'])
        df = df.drop('TeamID', axis=1).rename(columns={'Seed': 'SeedW'})
        df = pd.merge(df, df_seeds, how='left', left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID'])
        df = df.drop('TeamID', axis=1).rename(columns={'Seed': 'SeedL'})
        df['SeedW'] = df['SeedW'].apply(self._treat_seed)
        df['SeedL'] = df['SeedL'].apply(self._treat_seed)
        
        # Merge season-level features
        df = pd.merge(df, df_features_season, how='left', left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'])
        df = df.rename(columns={'WinRatio': 'WinRatioW', 'GapAvg': 'GapAvgW'}).drop('TeamID', axis=1)
        df = pd.merge(df, df_features_season, how='left', left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID'])
        df = df.rename(columns={'WinRatio': 'WinRatioL', 'GapAvg': 'GapAvgL'}).drop('TeamID', axis=1)
        
        # Create mirrored matches
        df = self.add_loosing_matches(df)
        
        # Compute difference features
        for col in ['Seed', 'WinRatio', 'GapAvg']:
            df[col + 'Diff'] = df[col + 'A'] - df[col + 'B']
        df['ScoreDiff'] = df['ScoreA'] - df['ScoreB']
        df['WinA'] = (df['ScoreDiff'] > 0).astype(int)
        
        self.df = df
        logging.info("Training data loaded and processed.")
    
    def load_test_data(self):
        """Load and prepare test data."""
        df_test = pd.read_csv(os.path.join(self.data_path, "SampleSubmissionStage1.csv"))
        df_test['Season'] = df_test['ID'].apply(lambda x: int(x.split('_')[0]))
        df_test['TeamIdA'] = df_test['ID'].apply(lambda x: int(x.split('_')[1]))
        df_test['TeamIdB'] = df_test['ID'].apply(lambda x: int(x.split('_')[2]))
        
        df_seeds = pd.concat([
            pd.read_csv(os.path.join(self.data_path, "MNCAATourneySeeds.csv")),
            pd.read_csv(os.path.join(self.data_path, "WNCAATourneySeeds.csv"))
        ], ignore_index=True)
        
        df_test = pd.merge(df_test, df_seeds, how='left', left_on=['Season', 'TeamIdA'], right_on=['Season', 'TeamID'])
        df_test = df_test.drop('TeamID', axis=1).rename(columns={'Seed': 'SeedA'}).fillna('W01')
        df_test = pd.merge(df_test, df_seeds, how='left', left_on=['Season', 'TeamIdB'], right_on=['Season', 'TeamID'])
        df_test = df_test.drop('TeamID', axis=1).rename(columns={'Seed': 'SeedB'}).fillna('W01')
        df_test['SeedA'] = df_test['SeedA'].apply(self._treat_seed)
        df_test['SeedB'] = df_test['SeedB'].apply(self._treat_seed)
        df_test = pd.merge(df_test, self.df_features_season, how='left', left_on=['Season', 'TeamIdA'], right_on=['Season', 'TeamID'])
        df_test = df_test.rename(columns={'WinRatio': 'WinRatioA', 'GapAvg': 'GapAvgA'}).drop('TeamID', axis=1)
        df_test = pd.merge(df_test, self.df_features_season, how='left', left_on=['Season', 'TeamIdB'], right_on=['Season', 'TeamID'])
        df_test = df_test.rename(columns={'WinRatio': 'WinRatioB', 'GapAvg': 'GapAvgB'}).drop('TeamID', axis=1)
        df_test["SeedDiff"] = df_test["SeedA"] - df_test["SeedB"]
        df_test["WinRatioDiff"] = df_test["WinRatioA"] - df_test["WinRatioB"]
        df_test["GapAvgDiff"] = df_test["GapAvgA"] - df_test["GapAvgB"]
        
        self.df_test = df_test
        logging.info("Test data loaded and processed.")
    
    def add_loosing_matches(self, df):
        """Create mirrored matches so that both teams appear as Team A."""
        win_rename = {
            "WTeamID": "TeamIdA", 
            "WScore": "ScoreA", 
            "LTeamID": "TeamIdB",
            "LScore": "ScoreB",
        }
        win_rename.update({c: c[:-1] + "A" for c in df.columns if c.endswith('W')})
        win_rename.update({c: c[:-1] + "B" for c in df.columns if c.endswith('L')})
        lose_rename = {
            "WTeamID": "TeamIdB", 
            "WScore": "ScoreB", 
            "LTeamID": "TeamIdA",
            "LScore": "ScoreA",
        }
        lose_rename.update({c: c[:-1] + "B" for c in df.columns if c.endswith('W')})
        lose_rename.update({c: c[:-1] + "A" for c in df.columns if c.endswith('L')})
        win_df = df.copy().rename(columns=win_rename)
        lose_df = df.copy().rename(columns=lose_rename)
        combined_df = pd.concat([win_df, lose_df], axis=0, sort=False)
        logging.info("Mirrored matches added.")
        return combined_df
    
    def gpu_target_encode(self, df, cat_cols, target):
        """Perform GPU target encoding using cuDF."""
        for col in cat_cols:
            df[col] = df[col].astype(str)
        global_mean = df[target].mean()
        mappings = {}
        for col in cat_cols:
            mapping = df.groupby(col)[target].mean().reset_index()
            mapping = mapping.rename(columns={target: col + '_target'})
            mappings[col] = mapping
            df = df.merge(mapping, on=col, how='left')
            df[col + '_target'] = df[col + '_target'].fillna(global_mean)
        logging.info("GPU target encoding completed.")
        return df, mappings
    
    def gpu_target_encode_test(self, df, mappings, cat_cols):
        """Apply target encoding mappings to test data."""
        for col in cat_cols:
            df[col] = df[col].astype(str)
            mapping = mappings[col]
            df = df.merge(mapping, on=col, how='left')
            global_mean = mapping[col + '_target'].mean()
            df[col + '_target'] = df[col + '_target'].fillna(global_mean)
        logging.info("GPU target encoding applied to test data.")
        return df
    
    def gpu_rescale(self, df, features):
        """Rescale features using GPU."""
        for col in features:
            df[col] = df[col].fillna(0.06250)
        mins = df[features].min()
        maxs = df[features].max()
        for col in features:
            diff = maxs[col] - mins[col]
            if diff == 0:
                df[col] = 0.0
            else:
                df[col] = (df[col] - mins[col]) / diff
        logging.info("GPU rescaling completed.")
        return df
    
    def prepare_gpu_data(self):
        """
        Convert training and test data to cuDF, perform target encoding and rescaling.
        Updates self.gdf and self.gdf_test.
        """
        self.features_gpu = self.features_numeric + [col + '_target' for col in self.cat_cols]
        gdf = cudf.DataFrame.from_pandas(self.df)
        gdf_test = cudf.DataFrame.from_pandas(self.df_test)
        gdf, self.mappings = self.gpu_target_encode(gdf, self.cat_cols, 'WinA')
        gdf_test = self.gpu_target_encode_test(gdf_test, self.mappings, self.cat_cols)
        gdf = gdf.reset_index(drop=True)
        gdf_test = gdf_test.reset_index(drop=True)
        gdf = self.gpu_rescale(gdf, self.features_gpu)
        gdf_test = self.gpu_rescale(gdf_test, self.features_gpu)
        self.gdf = gdf
        self.gdf_test = gdf_test
        logging.info("GPU data prepared.")
    
    def train_final_model(self):
        """
        Train a final CatBoost model on full training data.
        Returns X_train, y_train in Pandas and the trained model.
        """
        X_train = self.gdf[self.features_gpu].to_pandas()
        y_train = self.gdf['WinA'].to_pandas()
        model = CatBoostClassifier(**self.hyperparams)
        model.fit(X_train, y_train)
        self.final_model = model
        self.X_train = X_train
        self.y_train = y_train
        logging.info("Final CatBoost model trained.")
        return model
    
    def iterate_and_select_model(self, n_iter=7):
        """
        Save and reload the final model n_iter times, compute training MSE for each, and select the best.
        Returns the filename of the best model.
        """
        cv_mses = []
        for i in range(n_iter):
            model_filename = f"catboost_model_iter_{i}.cbm"
            self.final_model.save_model(model_filename)
            loaded_model = CatBoostClassifier()
            loaded_model.load_model(model_filename)
            preds = loaded_model.predict_proba(self.X_train)[:, 1]
            mse = ((self.y_train - preds) ** 2).mean()
            cv_mses.append(mse)
            logging.info(f"Iteration {i}: Training MSE = {mse:.4f}")
        best_iter = int(np.argmin(cv_mses))
        best_model_filename = f"catboost_model_iter_{best_iter}.cbm"
        logging.info(f"Best iteration: {best_iter} with Training MSE = {cv_mses[best_iter]:.4f}")
        self.best_model_filename = best_model_filename
        return best_model_filename
    
    def load_best_model(self):
        """Load the best saved model."""
        model = CatBoostClassifier()
        model.load_model(self.best_model_filename)
        self.best_model = model
        logging.info("Best model loaded.")
        return model
    
    def predict_test(self):
        """Generate test predictions using the best model."""
        X_test = self.gdf_test[self.features_gpu].to_pandas()
        preds_test = self.best_model.predict_proba(X_test)[:, 1]
        self.gdf_test['pred'] = preds_test
        final_sub = self.gdf_test[['ID', 'pred']].to_pandas()
        self.final_sub = final_sub
        logging.info("Test predictions generated.")
        return final_sub
    
    def generate_submission(self, filename='submission.csv'):
        """Save the submission file and plot the distribution."""
        self.final_sub.to_csv(filename, index=False)
        logging.info("Submission file generated.")
        print(self.final_sub.head())
        sns.displot(self.final_sub['pred'], kde=True)
        plt.title("Distribution of Final Predictions (GPU CatBoost)")
        plt.show()
    
    def run_pipeline(self):
        """
        Run the full pipeline.
        This method uses tqdm to display progress through the major steps.
        """
        steps = [
            ("Loading and preparing training data...", self.load_data),
            ("Loading test data...", self.load_test_data),
            ("Preparing GPU data (target encoding and rescaling)...", self.prepare_gpu_data),
            ("Training final CatBoost model on full data...", self.train_final_model),
            ("Iterating over saved models to select best iteration...", lambda: self.iterate_and_select_model(n_iter=7)),
            ("Loading best model...", self.load_best_model),
            ("Generating test predictions...", self.predict_test),
            ("Generating submission file...", self.generate_submission)
        ]
        for desc, func in tqdm(steps, desc="Pipeline Progress", total=len(steps)):
            tqdm.write(desc)
            logging.info(desc)
            func()




# Example usage:
if __name__ == "__main__":
    DATA_PATH='/kaggle/input/march-machine-learning-mania-2025'
    pipeline = CatBoostPipeline(DATA_PATH)
    #Optionally, adjust hyperparameters using a magic function:
    #pipeline.set_hyperparameters({"iterations": 600, "learning_rate": 0.04})
    pipeline.run_pipeline()



%%writefile catboostpipeline.py

import os, re, pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import logging
from sklearn.metrics import mean_squared_error
from tqdm import tqdm  # progress bar

# RAPIDS cuDF for GPU preprocessing
import cudf

# Import CatBoost (GPU-enabled)
from catboost import CatBoostClassifier, Pool

# Configure logging to output to "ml model metrics.txt"
logging.basicConfig(
    filename='ml_model_metrics.txt',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

class CatBoostPipeline:
    def __init__(self, data_path, hyperparams=None):
        """
        Initialize the pipeline.
        
        Parameters:
          data_path (str): Path to the data directory.
          hyperparams (dict): Dictionary of CatBoost hyperparameters.
        """
        self.data_path = data_path
        # Default hyperparameters if none provided:
        if hyperparams is None:
            self.hyperparams = {
                "iterations": 500,
                "learning_rate": 0.05,
                "depth": 6,
                "task_type": "GPU",
                "devices": '0',
                "verbose": 100
            }
        else:
            self.hyperparams = hyperparams
        
        # Feature definitions
        self.features_numeric = ['SeedA', 'SeedB', 'WinRatioA', 'GapAvgA', 
                                 'WinRatioB', 'GapAvgB', 'SeedDiff', 'WinRatioDiff', 'GapAvgDiff']
        self.cat_cols = ['TeamIdA', 'TeamIdB']
        # For GPU rescaling/target encoding we build a feature list that includes target-encoded columns.
        self.features_gpu = self.features_numeric + [col + '_target' for col in self.cat_cols]
    
    def set_hyperparameters(self, new_params):
        """Update hyperparameters."""
        self.hyperparams.update(new_params)
    
    def _treat_seed(self, seed):
        return int(re.sub("[^0-9]", "", str(seed)))
    
    def load_data(self):
        """Read seeds, regular season and tournament results; compute season-level features; merge seeds; create mirrored matches; compute difference features."""
        # Read tournament seeds
        df_seeds = pd.concat([
            pd.read_csv(os.path.join(self.data_path, "MNCAATourneySeeds.csv")),
            pd.read_csv(os.path.join(self.data_path, "WNCAATourneySeeds.csv"))
        ], ignore_index=True)
        
        # Read regular season results
        df_season_results = pd.concat([
            pd.read_csv(os.path.join(self.data_path, "MRegularSeasonCompactResults.csv")),
            pd.read_csv(os.path.join(self.data_path, "WRegularSeasonCompactResults.csv"))
        ], ignore_index=True)
        df_season_results.drop(['NumOT', 'WLoc'], axis=1, inplace=True)
        df_season_results['ScoreGap'] = df_season_results['WScore'] - df_season_results['LScore']
        
        # Compute season-level features
        num_win = (df_season_results.groupby(['Season', 'WTeamID'])
                   .count().reset_index()[['Season', 'WTeamID', 'DayNum']]
                   .rename(columns={"DayNum": "NumWins", "WTeamID": "TeamID"}))
        num_loss = (df_season_results.groupby(['Season', 'LTeamID'])
                    .count().reset_index()[['Season', 'LTeamID', 'DayNum']]
                    .rename(columns={"DayNum": "NumLosses", "LTeamID": "TeamID"}))
        gap_win = (df_season_results.groupby(['Season', 'WTeamID'])
                   .mean().reset_index()[['Season', 'WTeamID', 'ScoreGap']]
                   .rename(columns={"ScoreGap": "GapWins", "WTeamID": "TeamID"}))
        gap_loss = (df_season_results.groupby(['Season', 'LTeamID'])
                    .mean().reset_index()[['Season', 'LTeamID', 'ScoreGap']]
                    .rename(columns={"ScoreGap": "GapLosses", "LTeamID": "TeamID"}))
        df_features_season = pd.concat([
            df_season_results.groupby(['Season', 'WTeamID']).count().reset_index()[['Season', 'WTeamID']].rename(columns={"WTeamID": "TeamID"}),
            df_season_results.groupby(['Season', 'LTeamID']).count().reset_index()[['Season', 'LTeamID']].rename(columns={"LTeamID": "TeamID"})
        ], axis=0).drop_duplicates().sort_values(['Season', 'TeamID']).reset_index(drop=True)
        df_features_season = df_features_season.merge(num_win, on=['Season', 'TeamID'], how='left')
        df_features_season = df_features_season.merge(num_loss, on=['Season', 'TeamID'], how='left')
        df_features_season = df_features_season.merge(gap_win, on=['Season', 'TeamID'], how='left')
        df_features_season = df_features_season.merge(gap_loss, on=['Season', 'TeamID'], how='left')
        df_features_season.fillna(0, inplace=True)
        df_features_season['WinRatio'] = df_features_season['NumWins'] / (df_features_season['NumWins'] + df_features_season['NumLosses'])
        df_features_season['GapAvg'] = ((df_features_season['NumWins'] * df_features_season['GapWins'] - 
                                         df_features_season['NumLosses'] * df_features_season['GapLosses']) /
                                        (df_features_season['NumWins'] + df_features_season['NumLosses']))
        df_features_season.drop(['NumWins', 'NumLosses', 'GapWins', 'GapLosses'], axis=1, inplace=True)
        self.df_features_season = df_features_season
        
        # Read tournament results
        df_tourney_results = pd.concat([
            pd.read_csv(os.path.join(self.data_path, "WNCAATourneyCompactResults.csv")),
            pd.read_csv(os.path.join(self.data_path, "MNCAATourneyCompactResults.csv"))
        ], ignore_index=True)
        df_tourney_results.drop(['NumOT', 'WLoc'], axis=1, inplace=True)
        df = df_tourney_results.copy()
        df = df[df['Season'] >= 2016].reset_index(drop=True)
        
        # Merge seeds
        df = pd.merge(df, df_seeds, how='left', left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'])
        df = df.drop('TeamID', axis=1).rename(columns={'Seed': 'SeedW'})
        df = pd.merge(df, df_seeds, how='left', left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID'])
        df = df.drop('TeamID', axis=1).rename(columns={'Seed': 'SeedL'})
        df['SeedW'] = df['SeedW'].apply(self._treat_seed)
        df['SeedL'] = df['SeedL'].apply(self._treat_seed)
        
        # Merge season-level features
        df = pd.merge(df, df_features_season, how='left', left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'])
        df = df.rename(columns={'WinRatio': 'WinRatioW', 'GapAvg': 'GapAvgW'}).drop('TeamID', axis=1)
        df = pd.merge(df, df_features_season, how='left', left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID'])
        df = df.rename(columns={'WinRatio': 'WinRatioL', 'GapAvg': 'GapAvgL'}).drop('TeamID', axis=1)
        
        # Create mirrored matches
        df = self.add_loosing_matches(df)
        
        # Compute difference features
        for col in ['Seed', 'WinRatio', 'GapAvg']:
            df[col + 'Diff'] = df[col + 'A'] - df[col + 'B']
        df['ScoreDiff'] = df['ScoreA'] - df['ScoreB']
        df['WinA'] = (df['ScoreDiff'] > 0).astype(int)
        
        self.df = df
        logging.info("Training data loaded and processed.")
    
    def load_test_data(self):
        """Load and prepare test data."""
        df_test = pd.read_csv(os.path.join(self.data_path, "SampleSubmissionStage1.csv"))
        df_test['Season'] = df_test['ID'].apply(lambda x: int(x.split('_')[0]))
        df_test['TeamIdA'] = df_test['ID'].apply(lambda x: int(x.split('_')[1]))
        df_test['TeamIdB'] = df_test['ID'].apply(lambda x: int(x.split('_')[2]))
        
        df_seeds = pd.concat([
            pd.read_csv(os.path.join(self.data_path, "MNCAATourneySeeds.csv")),
            pd.read_csv(os.path.join(self.data_path, "WNCAATourneySeeds.csv"))
        ], ignore_index=True)
        
        df_test = pd.merge(df_test, df_seeds, how='left', left_on=['Season', 'TeamIdA'], right_on=['Season', 'TeamID'])
        df_test = df_test.drop('TeamID', axis=1).rename(columns={'Seed': 'SeedA'}).fillna('W01')
        df_test = pd.merge(df_test, df_seeds, how='left', left_on=['Season', 'TeamIdB'], right_on=['Season', 'TeamID'])
        df_test = df_test.drop('TeamID', axis=1).rename(columns={'Seed': 'SeedB'}).fillna('W01')
        df_test['SeedA'] = df_test['SeedA'].apply(self._treat_seed)
        df_test['SeedB'] = df_test['SeedB'].apply(self._treat_seed)
        df_test = pd.merge(df_test, self.df_features_season, how='left', left_on=['Season', 'TeamIdA'], right_on=['Season', 'TeamID'])
        df_test = df_test.rename(columns={'WinRatio': 'WinRatioA', 'GapAvg': 'GapAvgA'}).drop('TeamID', axis=1)
        df_test = pd.merge(df_test, self.df_features_season, how='left', left_on=['Season', 'TeamIdB'], right_on=['Season', 'TeamID'])
        df_test = df_test.rename(columns={'WinRatio': 'WinRatioB', 'GapAvg': 'GapAvgB'}).drop('TeamID', axis=1)
        df_test["SeedDiff"] = df_test["SeedA"] - df_test["SeedB"]
        df_test["WinRatioDiff"] = df_test["WinRatioA"] - df_test["WinRatioB"]
        df_test["GapAvgDiff"] = df_test["GapAvgA"] - df_test["GapAvgB"]
        
        self.df_test = df_test
        logging.info("Test data loaded and processed.")
    
    def add_loosing_matches(self, df):
        """Create mirrored matches so that both teams appear as Team A."""
        win_rename = {
            "WTeamID": "TeamIdA", 
            "WScore": "ScoreA", 
            "LTeamID": "TeamIdB",
            "LScore": "ScoreB",
        }
        win_rename.update({c: c[:-1] + "A" for c in df.columns if c.endswith('W')})
        win_rename.update({c: c[:-1] + "B" for c in df.columns if c.endswith('L')})
        lose_rename = {
            "WTeamID": "TeamIdB", 
            "WScore": "ScoreB", 
            "LTeamID": "TeamIdA",
            "LScore": "ScoreA",
        }
        lose_rename.update({c: c[:-1] + "B" for c in df.columns if c.endswith('W')})
        lose_rename.update({c: c[:-1] + "A" for c in df.columns if c.endswith('L')})
        win_df = df.copy().rename(columns=win_rename)
        lose_df = df.copy().rename(columns=lose_rename)
        combined_df = pd.concat([win_df, lose_df], axis=0, sort=False)
        logging.info("Mirrored matches added.")
        return combined_df
    
    def gpu_target_encode(self, df, cat_cols, target):
        """Perform GPU target encoding using cuDF."""
        for col in cat_cols:
            df[col] = df[col].astype(str)
        global_mean = df[target].mean()
        mappings = {}
        for col in cat_cols:
            mapping = df.groupby(col)[target].mean().reset_index()
            mapping = mapping.rename(columns={target: col + '_target'})
            mappings[col] = mapping
            df = df.merge(mapping, on=col, how='left')
            df[col + '_target'] = df[col + '_target'].fillna(global_mean)
        logging.info("GPU target encoding completed.")
        return df, mappings
    
    def gpu_target_encode_test(self, df, mappings, cat_cols):
        """Apply target encoding mappings to test data."""
        for col in cat_cols:
            df[col] = df[col].astype(str)
            mapping = mappings[col]
            df = df.merge(mapping, on=col, how='left')
            global_mean = mapping[col + '_target'].mean()
            df[col + '_target'] = df[col + '_target'].fillna(global_mean)
        logging.info("GPU target encoding applied to test data.")
        return df
    
    # Updated GPU rescale function as requested.
    def gpu_rescale(self, df, features):
        for col in features:
            df[col] = df[col].fillna(0.06250)
        mins = df[features].min()
        maxs = df[features].max()
        for col in features:
            diff = maxs[col] - mins[col]
            if diff == 0:
                df[col] = 0.0
            else:
                df[col] = (df[col] - mins[col]) / diff
        logging.info("GPU rescaling completed.")
        return df
    
    def prepare_gpu_data(self):
        """
        Convert training and test data to cuDF, perform target encoding and rescaling.
        Updates self.gdf and self.gdf_test.
        """
        # Build GPU feature list: numeric features + target-encoded categorical columns.
        self.features_gpu = self.features_numeric + [col + '_target' for col in self.cat_cols]
        # Convert to cuDF
        gdf = cudf.DataFrame.from_pandas(self.df)
        gdf_test = cudf.DataFrame.from_pandas(self.df_test)
        # Apply target encoding on training data
        gdf, self.mappings = self.gpu_target_encode(gdf, self.cat_cols, 'WinA')
        # Apply same encoding on test data
        gdf_test = self.gpu_target_encode_test(gdf_test, self.mappings, self.cat_cols)
        gdf = gdf.reset_index(drop=True)
        gdf_test = gdf_test.reset_index(drop=True)
        # Rescale numeric features robustly using the updated function.
        gdf = self.gpu_rescale(gdf, self.features_gpu)
        gdf_test = self.gpu_rescale(gdf_test, self.features_gpu)
        self.gdf = gdf
        self.gdf_test = gdf_test
        logging.info("GPU data prepared.")
    
    def train_final_model(self):
        """
        Train a final CatBoost model on full training data.
        Returns X_train, y_train in Pandas and the trained model.
        """
        X_train = self.gdf[self.features_gpu].to_pandas()
        y_train = self.gdf['WinA'].to_pandas()
        model = CatBoostClassifier(**self.hyperparams)
        model.fit(X_train, y_train)
        self.final_model = model
        self.X_train = X_train
        self.y_train = y_train
        logging.info("Final CatBoost model trained.")
        return model
    
    def iterate_and_select_model(self, n_iter=7):
        """
        Save and reload the final model n_iter times, compute training MSE for each, and select the best.
        Returns the filename of the best model.
        """
        cv_mses = []
        for i in range(n_iter):
            model_filename = f"catboost_model_iter_{i}.cbm"
            self.final_model.save_model(model_filename)
            loaded_model = CatBoostClassifier()
            loaded_model.load_model(model_filename)
            preds = loaded_model.predict_proba(self.X_train)[:, 1]
            mse = ((self.y_train - preds) ** 2).mean()
            cv_mses.append(mse)
            logging.info(f"Iteration {i}: Training MSE = {mse:.4f}")
        best_iter = int(np.argmin(cv_mses))
        best_model_filename = f"catboost_model_iter_{best_iter}.cbm"
        logging.info(f"Best iteration: {best_iter} with Training MSE = {cv_mses[best_iter]:.4f}")
        self.best_model_filename = best_model_filename
        return best_model_filename
    
    def load_best_model(self):
        """Load the best saved model."""
        model = CatBoostClassifier()
        model.load_model(self.best_model_filename)
        self.best_model = model
        logging.info("Best model loaded.")
        return model
    
    def predict_test(self):
        """Generate test predictions using the best model."""
        X_test = self.gdf_test[self.features_gpu].to_pandas()
        preds_test = self.best_model.predict_proba(X_test)[:, 1]
        self.gdf_test['pred'] = preds_test
        final_sub = self.gdf_test[['ID', 'pred']].to_pandas()
        self.final_sub = final_sub
        logging.info("Test predictions generated.")
        return final_sub
    
    def generate_submission(self, filename='submission.csv'):
        """Save the submission file and plot the distribution."""
        self.final_sub.to_csv(filename, index=False)
        logging.info("Submission file generated.")
        print(self.final_sub.head())
        sns.displot(self.final_sub['pred'], kde=True)
        plt.title("Distribution of Final Predictions (GPU CatBoost)")
        plt.show()
    
    def run_pipeline(self):
        """
        Run the full pipeline.
        This method uses tqdm to display progress through the major steps.
        """
        steps = [
            ("Loading and preparing training data...", self.load_data),
            ("Loading test data...", self.load_test_data),
            ("Preparing GPU data (target encoding and rescaling)...", self.prepare_gpu_data),
            ("Training final CatBoost model on full data...", self.train_final_model),
            ("Iterating over saved models to select best iteration...", lambda: self.iterate_and_select_model(n_iter=7)),
            ("Loading best model...", self.load_best_model),
            ("Generating test predictions...", self.predict_test),
            ("Generating submission file...", self.generate_submission)
        ]
        for desc, func in tqdm(steps, desc="Pipeline Progress", total=len(steps)):
            tqdm.write(desc)
            logging.info(desc)
            func()



# Move all necessary files
!cp -r /kaggle/input/aiagenticoptimizedcodellm/other/default/1/* /kaggle/working/
import sys 
sys.path.append('/kaggle/working/AIAgenticOptimizedCodeLLM.py')


import os
from kaggle_secrets import UserSecretsClient
from google import genai
from IPython.display import display, HTML
from AIAgenticOptimizedCodeLLM import AIAgenticOptimizedCodeLLM 


# -------------------------------------------------------------------
# Main execution
# -------------------------------------------------------------------
if __name__ == "__main__":
    # Get the API key using Kaggle's secrets (adjust as needed for your environment)
    user_secrets = UserSecretsClient()
    api_key = user_secrets.get_secret("GOOGLE_API_KEY")

    # Define file paths for metrics, original code, and the output HTML file.
    metrics_file_path = "/kaggle/working/ml_model_metrics.txt" ## pass ml model metrics text file
    code_file_path = "/kaggle/working/catboostpipeline.py" ## pass your current/previous py code file
    optimized_html_file_path = "/kaggle/working/catboostpipeline_v1_optimized.html" ## optimized ml model suggestion and new optimized code also available here

    # User prompt for improvements
    user_prompt = """
Please provide ways to improve the ML model metrics and new suggestion with optimized code to generalize the ml model that avoids overfitting and underfitting!!

in suggestions share me simple code snippet changes that will improve the standard ml model as well as optimized code !! increase more length responses
"""

    optimizer = AIAgenticOptimizedCodeLLM(api_key, metrics_file_path, code_file_path, optimized_html_file_path)
    html_response = optimizer.optimize_model_code(user_prompt)

    display(HTML(html_response))

