!pip install lightgbm
# %% [code]
# Install LightGBM if not installed (uncomment if needed)
# !pip install lightgbm


import numpy as np
import pandas as pd
import optuna
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import logging
import joblib
from datetime import datetime

# Configuration Class

class Config:
    DATA_PATHS = {
        'train': 'path/to/train.csv',
        'test': 'path/to/test.csv'
    }

class Config:
    RANDOM_STATE = 42
    N_FOLDS = 5
    OPTUNA_TRIALS = 50
    DATA_PATHS = {
        'train': '/kaggle/input/playground-series-s5e4/train.csv',
        'test': '/kaggle/input/playground-series-s5e4/test.csv',
        'submission': 'submission.csv'
    }
    LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'

# Initialize Logging
logging.basicConfig(level=logging.INFO, format=Config.LOG_FORMAT)
logger = logging.getLogger(__name__)




class DataLoader:
    """Handles data loading and initial preprocessing"""
    def __init__(self):
        self.train = None
        self.test = None
        
    def load(self):
        logger.info("Loading datasets...")
        self.train = pd.read_csv(Config.DATA_PATHS['train'], parse_dates=['Publication_Date'])
        self.test = pd.read_csv(Config.DATA_PATHS['test'], parse_dates=['Publication_Date'])
        self._extract_temporal_features()
        logger.info(f"Train shape: {self.train.shape}, Test shape: {self.test.shape}")
        return self.train, self.test
    
    def _extract_temporal_features(self):
        for df in [self.train, self.test]:
            df['Publication_Day'] = df['Publication_Date'].dt.day_name()
            df['Publication_Hour'] = df['Publication_Date'].dt.hour
            df['Publication_Quarter'] = df['Publication_Date'].dt.quarter



class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Advanced feature engineering transformer"""
    def __init__(self):
        self.feature_names = []
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Temporal Features
        X['Is_Weekend'] = X['Publication_Date'].dt.weekday >= 5
        X['Hour_Cluster'] = pd.cut(X['Publication_Hour'],
                                 bins=[0, 6, 12, 18, 24],
                                 labels=['Night', 'Morning', 'Afternoon', 'Evening'])
        
        # Interaction Features
        X['Popularity_Score'] = 0.6*X['Host_Popularity_percentage'] + 0.4*X['Guest_Popularity_percentage']
        X['Ad_Density'] = X['Number_of_Ads'] / X['Episode_Length_minutes'].clip(lower=1)
        
        # Advanced Metrics
        X['Length_Adjusted_Popularity'] = X['Popularity_Score'] * np.log1p(X['Episode_Length_minutes'])
        X['Time_Since_Publication'] = (datetime.now() - X['Publication_Date']).dt.days
        
        self.feature_names = X.columns.tolist()
        return X



def build_preprocessing_pipeline():
    """Constructs complete data processing pipeline"""
    numeric_features = ['Episode_Length_minutes', 'Host_Popularity_percentage',
                       'Guest_Popularity_percentage', 'Number_of_Ads']
    categorical_features = ['Podcast_Name', 'Genre', 'Hour_Cluster', 'Publication_Day']
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', FunctionTransformer(np.log1p, validate=True))
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse=False))
    ])
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])
    
    return Pipeline(steps=[
        ('feature_engineering', FeatureEngineer()),
        ('preprocessing', preprocessor)
    ])


class OptunaOptimizer:
    """Handles hyperparameter tuning with Optuna"""
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.study = None
        
    def objective(self, trial):
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'verbosity': -1,
            'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart']),
            'num_leaves': trial.suggest_int('num_leaves', 20, 300),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.3, log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 5, 100),
        }
        
        model = lgb.LGBMRegressor(**params, random_state=Config.RANDOM_STATE)
        cv = KFold(n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.RANDOM_STATE)
        scores = -cross_val_score(model, self.X, self.y, cv=cv, 
                                scoring='neg_root_mean_squared_error')
        return np.mean(scores)
    
    def optimize(self):
        logger.info("Starting hyperparameter optimization...")
        self.study = optuna.create_study(direction='minimize')
        self.study.optimize(self.objective, n_trials=Config.OPTUNA_TRIALS)
        logger.info(f"Best RMSE: {self.study.best_value:.4f}")
        return self.study.best_params



class LGBMModel:
    """Handles LightGBM model lifecycle"""
    def __init__(self, best_params=None):
        self.model = None
        self.best_params = best_params or {}
        
    def train(self, X, y):
        logger.info("Training final model...")
        self.model = lgb.LGBMRegressor(
            **self.best_params,
            random_state=Config.RANDOM_STATE
        )
        self.model.fit(X, y)
        
    def evaluate(self, X_val, y_val):
        preds = self.model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        logger.info(f"Validation RMSE: {rmse:.4f}")
        return rmse
    
    def get_feature_importance(self, feature_names):
        return pd.DataFrame({
            'feature': feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)


def main():
    # Data Loading
    loader = DataLoader()
    train, test = loader.load()
    
    # Pipeline Construction
    pipeline = build_preprocessing_pipeline()
    X_train = pipeline.fit_transform(train)
    y_train = train['Listening_Time_minutes']
    X_test = pipeline.transform(test)
    
    # Hyperparameter Tuning
    optimizer = OptunaOptimizer(X_train, y_train)
    best_params = optimizer.optimize()
    
    # Model Training
    model = LGBMModel(best_params)
    model.train(X_train, y_train)
    
    # Feature Analysis
    feature_names = pipeline.named_steps['preprocessing'].get_feature_names_out()
    importance = model.get_feature_importance(feature_names)
    logger.info("\nTop 10 Features:\n" + str(importance.head(10)))
    
    # Generate Submission
    test_preds = model.model.predict(X_test)
    submission = pd.DataFrame({
        'id': test['id'],
        'Listening_Time_minutes': test_preds
    })
    submission.to_csv(Config.DATA_PATHS['submission'], index=False)
    logger.info(f"Submission saved to {Config.DATA_PATHS['submission']}")
 




