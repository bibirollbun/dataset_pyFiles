import os

import warnings
warnings.filterwarnings('ignore')

DATA_PATH = '/kaggle/input/march-machine-learning-mania-2025/'

# data_loader.py
import pandas as pd
import re

class DataLoader:
    @staticmethod
    def load_seeds():
        return pd.concat([
            pd.read_csv(os.path.join(DATA_PATH, "MNCAATourneySeeds.csv")),
            pd.read_csv(os.path.join(DATA_PATH, "WNCAATourneySeeds.csv")),
        ], ignore_index=True)
    
    @staticmethod
    def load_season_results():
        df = pd.concat([
            pd.read_csv(os.path.join(DATA_PATH, "MRegularSeasonCompactResults.csv")),
            pd.read_csv(os.path.join(DATA_PATH, "WRegularSeasonCompactResults.csv")),
        ], ignore_index=True)
        
        df.drop(['NumOT', 'WLoc'], axis=1, inplace=True)
        df['ScoreGap'] = df['WScore'] - df['LScore']
        return df
    
    @staticmethod
    def load_tourney_results():
        return pd.concat([
            pd.read_csv(os.path.join(DATA_PATH, "WNCAATourneyCompactResults.csv")),
            pd.read_csv(os.path.join(DATA_PATH, "MNCAATourneyCompactResults.csv")),
        ], ignore_index=True)

class FeatureEngineer:
    @staticmethod
    def create_season_features(df_season_results):
        # Win/Loss counts
        num_win = df_season_results.groupby(['Season', 'WTeamID']).count()
        num_win = num_win.reset_index()[['Season', 'WTeamID', 'DayNum']].rename(
            columns={"DayNum": "NumWins", "WTeamID": "TeamID"})
        
        num_loss = df_season_results.groupby(['Season', 'LTeamID']).count()
        num_loss = num_loss.reset_index()[['Season', 'LTeamID', 'DayNum']].rename(
            columns={"DayNum": "NumLosses", "LTeamID": "TeamID"})
        
        # Score gaps
        gap_win = df_season_results.groupby(['Season', 'WTeamID']).mean().reset_index()
        gap_win = gap_win[['Season', 'WTeamID', 'ScoreGap']].rename(
            columns={"ScoreGap": "GapWins", "WTeamID": "TeamID"})
        
        gap_loss = df_season_results.groupby(['Season', 'LTeamID']).mean().reset_index()
        gap_loss = gap_loss[['Season', 'LTeamID', 'ScoreGap']].rename(
            columns={"ScoreGap": "GapLosses", "LTeamID": "TeamID"})
        
        # Combine features
        df_features = FeatureEngineer._get_base_features(df_season_results)
        df_features = FeatureEngineer._merge_features(df_features, [num_win, num_loss, gap_win, gap_loss])
        
        return FeatureEngineer._calculate_ratios(df_features)
    
    @staticmethod
    def _get_base_features(df_season_results):
        df_w = df_season_results.groupby(['Season', 'WTeamID']).count().reset_index()[
            ['Season', 'WTeamID']].rename(columns={"WTeamID": "TeamID"})
        df_l = df_season_results.groupby(['Season', 'LTeamID']).count().reset_index()[
            ['Season', 'LTeamID']].rename(columns={"LTeamID": "TeamID"})
        return pd.concat([df_w, df_l], axis=0).drop_duplicates().sort_values(
            ['Season', 'TeamID']).reset_index(drop=True)
    
    @staticmethod
    def _merge_features(df_features, feature_dfs):
        for feature_df in feature_dfs:
            df_features = df_features.merge(feature_df, on=['Season', 'TeamID'], how='left')
        return df_features.fillna(0)
    
    @staticmethod
    def _calculate_ratios(df):
        df['WinRatio'] = df['NumWins'] / (df['NumWins'] + df['NumLosses'])
        df['GapAvg'] = (
            (df['NumWins'] * df['GapWins'] - df['NumLosses'] * df['GapLosses'])
            / (df['NumWins'] + df['NumLosses'])
        )
        return df.drop(['NumWins', 'NumLosses', 'GapWins', 'GapLosses'], axis=1)

class DataProcessor:
    @staticmethod
    def treat_seed(seed):
        return int(re.sub("[^0-9]", "", seed))
    
    @staticmethod
    def process_tournament_data(df_tourney, df_seeds, df_features, min_season=2016):
        df = df_tourney[df_tourney['Season'] >= min_season].copy()
        df = DataProcessor._merge_seeds(df, df_seeds)
        df = DataProcessor._merge_features(df, df_features)
        df = DataProcessor.add_losing_matches(df)
        return DataProcessor._add_difference_features(df)
    
    @staticmethod
    def _merge_seeds(df, df_seeds):
        df = df.merge(
            df_seeds,
            how='left',
            left_on=['Season', 'WTeamID'],
            right_on=['Season', 'TeamID']
        ).drop('TeamID', axis=1).rename(columns={'Seed': 'SeedW'})
        
        df = df.merge(
            df_seeds,
            how='left',
            left_on=['Season', 'LTeamID'],
            right_on=['Season', 'TeamID']
        ).drop('TeamID', axis=1).rename(columns={'Seed': 'SeedL'})
        
        df['SeedW'] = df['SeedW'].apply(DataProcessor.treat_seed)
        df['SeedL'] = df['SeedL'].apply(DataProcessor.treat_seed)
        return df
    
    @staticmethod
    def _merge_features(df, df_features):
        # Merge winning team features
        df = df.merge(
            df_features,
            how='left',
            left_on=['Season', 'WTeamID'],
            right_on=['Season', 'TeamID']
        ).rename(columns={
            'WinRatio': 'WinRatioW',
            'GapAvg': 'GapAvgW',
        }).drop(columns='TeamID', axis=1)
        
        # Merge losing team features
        df = df.merge(
            df_features,
            how='left',
            left_on=['Season', 'LTeamID'],
            right_on=['Season', 'TeamID']
        ).rename(columns={
            'WinRatio': 'WinRatioL',
            'GapAvg': 'GapAvgL',
        }).drop(columns='TeamID', axis=1)
        
        return df
    
    @staticmethod
    def add_losing_matches(df):
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
        
        win_df = df.rename(columns=win_rename)
        lose_df = df.rename(columns=lose_rename)
        
        return pd.concat([win_df, lose_df], axis=0, sort=False)
    
    @staticmethod
    def _add_difference_features(df):
        cols_to_diff = ['Seed', 'WinRatio', 'GapAvg']
        for col in cols_to_diff:
            df[col + 'Diff'] = df[col + 'A'] - df[col + 'B']
        
        df['ScoreDiff'] = df['ScoreA'] - df['ScoreB']
        df['WinA'] = (df['ScoreDiff'] > 0).astype(int)
        return df

from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import ExtraTreesClassifier
import numpy as np

class ModelTrainer:
    def __init__(self, features):
        self.features = features
        self.model = ExtraTreesClassifier()
        self.scaler = MinMaxScaler()
    
    def train_and_evaluate(self, df_train, df_val, df_test=None):
        if df_test is not None:
            X_train, X_val, X_test = self._prepare_data(df_train, df_val, df_test)
        else:
            X_train, X_val = self._prepare_data(df_train, df_val)
            
        y_train = df_train['WinA']
        
        self.model.fit(X_train, y_train)
        predictions = self.model.predict_proba(X_val)[:, 1]
        
        if df_test is not None:
            test_predictions = self.model.predict_proba(X_test)[:, 1]
            return predictions, test_predictions
        return predictions
    
    def _prepare_data(self, df_train, df_val, df_test=None):
        # Fill missing values in training and validation sets
        for df in [df_train, df_val]:
            for col in self.features:
                df[col].fillna(df_train[col].mean(), inplace=True)
        
        # Fit scaler on training data and transform all datasets
        self.scaler.fit(df_train[self.features])
        
        X_train = self.scaler.transform(df_train[self.features])
        X_val = self.scaler.transform(df_val[self.features])
        
        if df_test is not None:
            # Fill missing values in test set using training means
            for col in self.features:
                df_test[col].fillna(df_train[col].mean(), inplace=True)
            
            X_test = self.scaler.transform(df_test[self.features])
            return X_train, X_val, X_test
        return X_train, X_val

def prepare_test_data(data_path, df_seeds, df_features):
    # Read submission format
    df_test = pd.read_csv(os.path.join(data_path, "SampleSubmissionStage1.csv"))
    
    # Extract season and team IDs
    df_test['Season'] = df_test['ID'].apply(lambda x: int(x.split('_')[0]))
    df_test['TeamIdA'] = df_test['ID'].apply(lambda x: int(x.split('_')[1]))
    df_test['TeamIdB'] = df_test['ID'].apply(lambda x: int(x.split('_')[2]))
    
    # Merge seeds
    df_test = df_test.merge(
        df_seeds,
        how='left',
        left_on=['Season', 'TeamIdA'],
        right_on=['Season', 'TeamID']
    ).drop('TeamID', axis=1).rename(columns={'Seed': 'SeedA'}).fillna('W01')
    
    df_test = df_test.merge(
        df_seeds,
        how='left',
        left_on=['Season', 'TeamIdB'],
        right_on=['Season', 'TeamID']
    ).drop('TeamID', axis=1).rename(columns={'Seed': 'SeedB'}).fillna('W01')
    
    # Process seeds
    df_test['SeedA'] = df_test['SeedA'].apply(DataProcessor.treat_seed)
    df_test['SeedB'] = df_test['SeedB'].apply(DataProcessor.treat_seed)
    
    # Merge features
    df_test = df_test.merge(
        df_features,
        how='left',
        left_on=['Season', 'TeamIdA'],
        right_on=['Season', 'TeamID']
    ).rename(columns={
        'WinRatio': 'WinRatioA',
        'GapAvg': 'GapAvgA',
    }).drop(columns='TeamID', axis=1)
    
    df_test = df_test.merge(
        df_features,
        how='left',
        left_on=['Season', 'TeamIdB'],
        right_on=['Season', 'TeamID']
    ).rename(columns={
        'WinRatio': 'WinRatioB',
        'GapAvg': 'GapAvgB',
    }).drop(columns='TeamID', axis=1)
    
    # Add difference features
    for col in ['Seed', 'WinRatio', 'GapAvg']:
        df_test[col + 'Diff'] = df_test[col + 'A'] - df_test[col + 'B']
    
    # Fill missing values
    for col in ['WinRatioA', 'WinRatioB', 'GapAvgA', 'GapAvgB', 'WinRatioDiff', 'GapAvgDiff']:
        df_test[col].fillna(df_test[col].mean(), inplace=True)
        
    return df_test

def main():
    # Load data
    data_loader = DataLoader()
    df_seeds = data_loader.load_seeds()
    df_season_results = data_loader.load_season_results()
    df_tourney_results = data_loader.load_tourney_results()
    
    # Feature engineering
    feature_engineer = FeatureEngineer()
    df_features = feature_engineer.create_season_features(df_season_results)
    
    # Process tournament data
    processor = DataProcessor()
    df = processor.process_tournament_data(df_tourney_results, df_seeds, df_features)
    
    # Prepare test data
    df_test = prepare_test_data(DATA_PATH, df_seeds, df_features)
    
    # Define features for model
    features = [
        "SeedA", "SeedB", 'WinRatioA', 'GapAvgA', 'WinRatioB', 'GapAvgB',
        'SeedDiff', 'WinRatioDiff', 'GapAvgDiff'
    ]
    
    # Train model using cross-validation
    seasons = df['Season'].unique()
    validation_scores = []
    test_predictions = []
    
    for season in seasons[1:]:
        print(f'Training for season {season}')
        df_train = df[df['Season'] < season].copy()
        df_val = df[df['Season'] == season].copy()
        
        trainer = ModelTrainer(features)
        val_predictions, test_pred = trainer.train_and_evaluate(df_train, df_val, df_test)
        
        # Calculate score for this validation fold
        score = ((df_val['WinA'].values - val_predictions) ** 2).mean()
        validation_scores.append(score)
        test_predictions.append(test_pred)
        print(f'\tValidation score: {score:.3f}')
    
    print(f'Average CV score: {np.mean(validation_scores):.3f}')
    
    # Generate submission file
    final_predictions = np.mean(test_predictions, axis=0)
    submission = pd.DataFrame({
        'ID': df_test['ID'],
        'Pred': final_predictions
    })
    submission.to_csv('submission.csv', index=False)
    print("\nSubmission file generated: submission.csv")

if __name__ == "__main__":
    main()

