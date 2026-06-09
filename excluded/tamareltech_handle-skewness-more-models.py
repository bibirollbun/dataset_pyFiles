#!/usr/bin/env python3
"""
Advanced Insurance Premium Prediction Pipeline
- Memory-efficient processing
- Chunk-based operations
- Genetic algorithm for feature selection
- Advanced ensemble methods
"""

import gc
import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Memory optimization
from pandas.api.types import CategoricalDtype
import psutil

# ML imports
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.metrics import mean_squared_log_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet, HuberRegressor
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

# Advanced optimization
import optuna
from deap import base, creator, tools, algorithms
import joblib
from joblib import Parallel, delayed

# Set random seed
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("="*100)
print("ADVANCED INSURANCE PREMIUM PREDICTION PIPELINE")
print("="*100)
print(f"Started at: {datetime.now()}")
print(f"Available RAM: {psutil.virtual_memory().available / 1024**3:.2f} GB")
print("="*100)

# ========================
# MEMORY OPTIMIZATION
# ========================

def optimize_dtypes(df, verbose=True):
    """Optimize dataframe dtypes to reduce memory usage"""
    if verbose:
        start_mem = df.memory_usage().sum() / 1024**2
        print(f"Memory usage before optimization: {start_mem:.2f} MB")
    
    # Optimize numeric columns
    for col in df.select_dtypes(include=['int']).columns:
        col_min = df[col].min()
        col_max = df[col].max()
        
        if col_min >= 0:
            if col_max < 255:
                df[col] = df[col].astype(np.uint8)
            elif col_max < 65535:
                df[col] = df[col].astype(np.uint16)
            elif col_max < 4294967295:
                df[col] = df[col].astype(np.uint32)
        else:
            if col_min > np.iinfo(np.int8).min and col_max < np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif col_min > np.iinfo(np.int16).min and col_max < np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif col_min > np.iinfo(np.int32).min and col_max < np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
    
    # Optimize float columns
    for col in df.select_dtypes(include=['float']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    
    # Convert object columns with low cardinality to category
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].nunique() / len(df) < 0.5:
            df[col] = df[col].astype('category')
    
    if verbose:
        end_mem = df.memory_usage().sum() / 1024**2
        print(f"Memory usage after optimization: {end_mem:.2f} MB")
        print(f"Memory reduction: {100 * (start_mem - end_mem) / start_mem:.1f}%")
    
    return df

# ========================
# DATA LOADING
# ========================

print("\nğŸ“Š LOADING DATA WITH MEMORY OPTIMIZATION...")

# Load data in chunks for memory efficiency
def load_data_optimized(filepath, chunksize=50000):
    """Load data in chunks and optimize dtypes"""
    chunks = []
    
    for chunk in pd.read_csv(filepath, chunksize=chunksize):
        chunk = optimize_dtypes(chunk, verbose=False)
        chunks.append(chunk)
    
    df = pd.concat(chunks, ignore_index=True)
    return df

# Load data
train_df = load_data_optimized('/kaggle/input/big-oai-final-course-1/train.csv')
test_df = load_data_optimized('/kaggle/input/big-oai-final-course-1/test.csv')

print(f"âœ… Train data: {train_df.shape[0]:,} rows Ã— {train_df.shape[1]} columns")
print(f"âœ… Test data: {test_df.shape[0]:,} rows Ã— {test_df.shape[1]} columns")

# Save test IDs
test_ids = test_df['id'].values

# ========================
# FEATURE ENGINEERING
# ========================

class AdvancedFeatureEngineer:
    """Advanced feature engineering with memory efficiency"""
    
    def __init__(self):
        self.date_features = []
        self.encoders = {}
        self.scalers = {}
        self.imputers = {}
    
    def process_dates(self, df):
        """Extract advanced date features"""
        df = df.copy()
        
        if 'Policy Start Date' in df.columns:
            df['Policy Start Date'] = pd.to_datetime(df['Policy Start Date'])
            
            # Basic features
            df['year'] = df['Policy Start Date'].dt.year.astype(np.int16)
            df['month'] = df['Policy Start Date'].dt.month.astype(np.int8)
            df['day'] = df['Policy Start Date'].dt.day.astype(np.int8)
            df['dayofweek'] = df['Policy Start Date'].dt.dayofweek.astype(np.int8)
            df['quarter'] = df['Policy Start Date'].dt.quarter.astype(np.int8)
            df['weekofyear'] = df['Policy Start Date'].dt.isocalendar().week.astype(np.int8)
            
            # Advanced features
            df['is_weekend'] = (df['dayofweek'] >= 5).astype(np.int8)
            df['is_month_start'] = (df['day'] <= 7).astype(np.int8)
            df['is_month_end'] = (df['day'] >= 24).astype(np.int8)
            df['is_quarter_start'] = ((df['month'] - 1) % 3 == 0) & (df['day'] <= 7)
            df['is_quarter_start'] = df['is_quarter_start'].astype(np.int8)
            
            # Cyclic encoding
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12).astype(np.float32)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12).astype(np.float32)
            df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31).astype(np.float32)
            df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31).astype(np.float32)
            
            # Days since start
            min_date = df['Policy Start Date'].min()
            df['days_since_start'] = (df['Policy Start Date'] - min_date).dt.days.astype(np.int16)
            
            df.drop('Policy Start Date', axis=1, inplace=True)
            
        return df
    
    def create_interaction_features(self, df):
        """Create advanced interaction features"""
        df = df.copy()
        
        # Age-based risk features
        if 'Age' in df.columns:
            df['age_risk'] = np.where(df['Age'] < 25, 2,
                            np.where(df['Age'] > 65, 2,
                            np.where(df['Age'] > 55, 1, 0))).astype(np.int8)
            
            # Age groups with better granularity
            df['age_group'] = pd.cut(df['Age'], 
                                    bins=[0, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 100],
                                    labels=['<20', '20-25', '25-30', '30-35', '35-40', 
                                           '40-45', '45-50', '50-55', '55-60', '60-65', '65+'])
        
        # Income features
        if 'Annual Income' in df.columns:
            # Log transform for skewed distribution
            df['income_log'] = np.log1p(df['Annual Income']).astype(np.float32)
            
            # Income per dependent
            if 'Number of Dependents' in df.columns:
                df['income_per_dependent'] = df['Annual Income'] / (df['Number of Dependents'] + 1)
                df['income_per_dependent'] = df['income_per_dependent'].astype(np.float32)
        
        # Health risk score
        if 'Health Score' in df.columns and 'Smoking Status' in df.columns:
            smoking_mult = df['Smoking Status'].map({'Yes': 1.5, 'No': 1.0})
            df['health_risk_adjusted'] = df['Health Score'] * smoking_mult
            df['health_risk_adjusted'] = df['health_risk_adjusted'].astype(np.float32)
        
        # Vehicle features
        if 'Vehicle Age' in df.columns:
            df['vehicle_risk'] = np.where(df['Vehicle Age'] > 15, 3,
                                 np.where(df['Vehicle Age'] > 10, 2,
                                 np.where(df['Vehicle Age'] > 5, 1, 0))).astype(np.int8)
        
        # Credit score categories
        if 'Credit Score' in df.columns:
            df['credit_category'] = pd.cut(df['Credit Score'],
                                          bins=[0, 580, 670, 740, 800, 1000],
                                          labels=['Poor', 'Fair', 'Good', 'VeryGood', 'Excellent'])
        
        # Policy features
        if 'Insurance Duration' in df.columns:
            df['is_new_customer'] = (df['Insurance Duration'] <= 1).astype(np.int8)
            df['is_loyal_customer'] = (df['Insurance Duration'] >= 5).astype(np.int8)
        
        # Exercise and health interaction
        if 'Exercise Frequency' in df.columns and 'Health Score' in df.columns:
            exercise_map = {'Never': 0, 'Monthly': 1, 'Weekly': 2, 'Daily': 3}
            exercise_score = df['Exercise Frequency'].map(exercise_map)
            df['fitness_score'] = exercise_score * df['Health Score'] / 3
            df['fitness_score'] = df['fitness_score'].astype(np.float32)
        
        return df
    
    def engineer_features(self, df):
        """Main feature engineering pipeline"""
        print("ğŸ”§ Engineering features...")
        
        # Process dates
        df = self.process_dates(df)
        
        # Create interactions
        df = self.create_interaction_features(df)
        
        # Optimize memory
        df = optimize_dtypes(df, verbose=False)
        
        return df

# ========================
# PREPROCESSING PIPELINE
# ========================

class MemoryEfficientPreprocessor:
    """Memory-efficient preprocessing with advanced imputation"""
    
    def __init__(self):
        self.numeric_features = []
        self.categorical_features = []
        self.target_encoder_features = []
        self.label_encoders = {}
        self.target_encoders = {}
        self.scalers = {}
        self.imputers = {}
        
    def identify_features(self, df, target_col=None):
        """Identify feature types"""
        if target_col and target_col in df.columns:
            df = df.drop(columns=[target_col])
        
        self.numeric_features = df.select_dtypes(include=['int8', 'int16', 'int32', 'int64', 
                                                          'uint8', 'uint16', 'uint32', 'uint64',
                                                          'float16', 'float32', 'float64']).columns.tolist()
        
        self.categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Remove ID columns
        for col in ['id', 'Id', 'ID']:
            if col in self.numeric_features:
                self.numeric_features.remove(col)
            if col in self.categorical_features:
                self.categorical_features.remove(col)
        
        # High cardinality features for target encoding
        self.target_encoder_features = [col for col in self.categorical_features 
                                       if df[col].nunique() > 10]
        
        print(f"ğŸ“Š Numeric features: {len(self.numeric_features)}")
        print(f"ğŸ“Š Categorical features: {len(self.categorical_features)}")
        print(f"ğŸ“Š Target encoding features: {len(self.target_encoder_features)}")
    
    def impute_missing(self, df, strategy='smart'):
        """Advanced missing value imputation"""
        df = df.copy()
        
        # Numeric imputation
        for col in self.numeric_features:
            if col in df.columns and df[col].isnull().any():
                if strategy == 'smart':
                    # Use different strategies based on missing percentage
                    missing_pct = df[col].isnull().sum() / len(df)
                    
                    if missing_pct < 0.05:
                        # KNN imputation for low missing percentage
                        imputer = KNNImputer(n_neighbors=5)
                        df[col] = imputer.fit_transform(df[[col]])
                    else:
                        # Median for higher missing percentage
                        df[col].fillna(df[col].median(), inplace=True)
                else:
                    df[col].fillna(df[col].median(), inplace=True)
        
        # Categorical imputation
        for col in self.categorical_features:
            if col in df.columns:
                df[col].fillna('Unknown', inplace=True)
        
        return df
    
    def encode_categoricals(self, df, y=None, is_train=True):
        """Encode categorical variables with multiple strategies"""
        df = df.copy()
        
        # Label encoding for ordinal features
        ordinal_features = ['Education Level', 'Exercise Frequency']
        ordinal_mappings = {
            'Education Level': {'High School': 0, "Bachelor's": 1, "Master's": 2, 'PhD': 3},
            'Exercise Frequency': {'Never': 0, 'Monthly': 1, 'Weekly': 2, 'Daily': 3}
        }
        
        for col in ordinal_features:
            if col in df.columns:
                if col in ordinal_mappings:
                    df[col] = df[col].map(ordinal_mappings[col])
                    df[col].fillna(0, inplace=True)
        
        # One-hot encoding for low cardinality
        low_card_features = [col for col in self.categorical_features 
                           if col not in self.target_encoder_features 
                           and col not in ordinal_features
                           and col in df.columns]
        
        if low_card_features:
            df = pd.get_dummies(df, columns=low_card_features, drop_first=True, dtype=np.int8)
        
        # Target encoding for high cardinality
        if y is not None and self.target_encoder_features:
            for col in self.target_encoder_features:
                if col in df.columns:
                    if is_train:
                        # Calculate target encoding with smoothing
                        smooth_factor = 10
                        mean_target = y.mean()
                        
                        encoding_dict = {}
                        for cat in df[col].unique():
                            mask = df[col] == cat
                            cat_count = mask.sum()
                            cat_mean = y[mask].mean() if mask.sum() > 0 else mean_target
                            
                            # Smoothing
                            smooth_mean = (cat_count * cat_mean + smooth_factor * mean_target) / (cat_count + smooth_factor)
                            encoding_dict[cat] = smooth_mean
                        
                        self.target_encoders[col] = encoding_dict
                        df[col + '_target_enc'] = df[col].map(encoding_dict).fillna(mean_target)
                    else:
                        # Use saved encodings for test set
                        if col in self.target_encoders:
                            mean_target = np.mean(list(self.target_encoders[col].values()))
                            df[col + '_target_enc'] = df[col].map(self.target_encoders[col]).fillna(mean_target)
                    
                    df.drop(columns=[col], inplace=True)
        
        return df
    
    def scale_features(self, df, is_train=True):
        """Scale numeric features"""
        df = df.copy()
        
        # Get numeric columns in the current dataframe
        numeric_cols = [col for col in self.numeric_features if col in df.columns]
        
        if numeric_cols:
            if is_train:
                self.scalers['robust'] = RobustScaler()
                df[numeric_cols] = self.scalers['robust'].fit_transform(df[numeric_cols])
            else:
                if 'robust' in self.scalers:
                    df[numeric_cols] = self.scalers['robust'].transform(df[numeric_cols])
        
        return df
    
    def preprocess(self, df, y=None, is_train=True):
        """Main preprocessing pipeline"""
        # Impute missing values
        df = self.impute_missing(df)
        
        # Encode categoricals
        df = self.encode_categoricals(df, y, is_train)
        
        # Scale features
        df = self.scale_features(df, is_train)
        
        # Final memory optimization
        df = optimize_dtypes(df, verbose=False)
        
        return df

# ========================
# GENETIC ALGORITHM FOR FEATURE SELECTION
# ========================

class GeneticFeatureSelector:
    """Genetic algorithm for optimal feature selection"""
    
    def __init__(self, n_features, n_generations=20, population_size=50):
        self.n_features = n_features
        self.n_generations = n_generations
        self.population_size = population_size
        self.best_features = None
        self.best_score = None
        
    def evaluate_features(self, individual, X, y, cv=3):
        """Evaluate a feature subset using cross-validation"""
        if sum(individual) == 0:
            return (1e10,)  # Return high error for empty feature set
        
        # Select features
        feature_mask = np.array(individual, dtype=bool)
        X_subset = X[:, feature_mask]
        
        # Use simple model for speed
        model = lgb.LGBMRegressor(
            n_estimators=50,
            num_leaves=31,
            learning_rate=0.1,
            verbosity=-1,
            random_state=RANDOM_STATE
        )
        
        # Cross-validation
        scores = cross_val_score(model, X_subset, y, cv=cv, 
                               scoring='neg_mean_squared_error', n_jobs=-1)
        
        return (-scores.mean(),)
    
    def run_genetic_algorithm(self, X, y):
        """Run genetic algorithm for feature selection"""
        print("ğŸ§¬ Running genetic algorithm for feature selection...")
        
        # Create fitness and individual classes
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)
        
        # Initialize toolbox
        toolbox = base.Toolbox()
        toolbox.register("attr_bool", np.random.randint, 0, 2)
        toolbox.register("individual", tools.initRepeat, creator.Individual,
                        toolbox.attr_bool, n=self.n_features)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        
        # Register genetic operators
        toolbox.register("evaluate", self.evaluate_features, X=X, y=y)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
        toolbox.register("select", tools.selTournament, tournsize=3)
        
        # Create initial population
        population = toolbox.population(n=self.population_size)
        
        # Run evolution
        for gen in range(self.n_generations):
            # Evaluate fitness
            fitnesses = Parallel(n_jobs=-1)(
                delayed(self.evaluate_features)(ind, X, y) for ind in population
            )
            for ind, fit in zip(population, fitnesses):
                ind.fitness.values = fit
            
            # Select next generation
            offspring = toolbox.select(population, len(population))
            offspring = list(map(toolbox.clone, offspring))
            
            # Apply crossover and mutation
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if np.random.random() < 0.5:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
            
            for mutant in offspring:
                if np.random.random() < 0.2:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values
            
            # Evaluate offspring
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = Parallel(n_jobs=-1)(
                delayed(self.evaluate_features)(ind, X, y) for ind in invalid_ind
            )
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            
            population[:] = offspring
            
            # Track best
            fits = [ind.fitness.values[0] for ind in population]
            best_idx = np.argmin(fits)
            
            if gen % 5 == 0:
                print(f"   Generation {gen}: Best fitness = {fits[best_idx]:.4f}")
        
        # Get best solution
        best_ind = tools.selBest(population, 1)[0]
        self.best_features = np.array(best_ind, dtype=bool)
        self.best_score = best_ind.fitness.values[0]
        
        print(f"âœ… Selected {sum(self.best_features)} features out of {self.n_features}")
        
        return self.best_features

# ========================
# ADVANCED ENSEMBLE MODEL
# ========================

class AdvancedEnsembleModel:
    """Advanced ensemble with stacking and blending"""
    
    def __init__(self):
        self.base_models = {}
        self.meta_model = None
        self.weights = None
        self.use_genetic_features = True
        self.feature_selector = None
        self.selected_features = None
        
    def initialize_models(self):
        """Initialize diverse base models"""
        self.base_models = {
            'lgb': lgb.LGBMRegressor(
                n_estimators=1000,
                num_leaves=31,
                learning_rate=0.05,
                feature_fraction=0.8,
                bagging_fraction=0.8,
                bagging_freq=5,
                verbosity=-1,
                random_state=RANDOM_STATE
            ),
            'xgb': xgb.XGBRegressor(
                n_estimators=1000,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=RANDOM_STATE
            ),
            'cat': CatBoostRegressor(
                iterations=1000,
                learning_rate=0.05,
                depth=6,
                verbose=False,
                random_state=RANDOM_STATE
            ),
            'rf': RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=RANDOM_STATE,
                n_jobs=-1
            ),
            'gb': GradientBoostingRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                random_state=RANDOM_STATE
            ),
            'huber': HuberRegressor(
                max_iter=200,
                alpha=1.0,
                epsilon=1.35
            )
        }
        
        # Meta model for stacking
        self.meta_model = Ridge(alpha=1.0)
    
    def train_with_cv(self, X_train, y_train, n_folds=5):
        """Train models with cross-validation"""
        print("ğŸ�¯ Training ensemble models...")
        
        # Genetic feature selection (optional)
        if self.use_genetic_features and X_train.shape[1] > 50:
            self.feature_selector = GeneticFeatureSelector(
                n_features=X_train.shape[1],
                n_generations=15,
                population_size=30
            )
            self.selected_features = self.feature_selector.run_genetic_algorithm(X_train, y_train)
            X_train = X_train[:, self.selected_features]
        
        # Initialize models
        self.initialize_models()
        
        # Prepare for stacking
        n_samples = X_train.shape[0]
        n_models = len(self.base_models)
        
        # OOF predictions for stacking
        oof_predictions = np.zeros((n_samples, n_models))
        
        # K-Fold cross-validation
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
        
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train)):
            print(f"\n  Fold {fold_idx + 1}/{n_folds}")
            
            X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
            y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
            
            # Train each base model
            for model_idx, (name, model) in enumerate(self.base_models.items()):
                print(f"    Training {name}...", end='')
                
                # Clone model to avoid fitting on same instance
                model_clone = model.__class__(**model.get_params())
                
                # Fit model
                if name in ['lgb', 'xgb', 'cat']:
                    model_clone.fit(
                        X_fold_train, y_fold_train,
                        eval_set=[(X_fold_val, y_fold_val)],
                        verbose=False
                    )
                else:
                    model_clone.fit(X_fold_train, y_fold_train)
                
                # Store OOF predictions
                oof_predictions[val_idx, model_idx] = model_clone.predict(X_fold_val)
                
                # Calculate fold score
                fold_score = np.sqrt(mean_squared_error(y_fold_val, oof_predictions[val_idx, model_idx]))
                print(f" RMSE: {fold_score:.4f}")
        
        # Train meta model on OOF predictions
        print("\n  Training meta model...")
        self.meta_model.fit(oof_predictions, y_train)
        
        # Calculate final OOF score
        final_predictions = self.meta_model.predict(oof_predictions)
        final_score = np.sqrt(mean_squared_error(y_train, final_predictions))
        print(f"\nâœ… Ensemble OOF RMSE: {final_score:.4f}")
        
        # Retrain all models on full data
        print("\n  Retraining on full dataset...")
        for name, model in self.base_models.items():
            if name in ['lgb', 'xgb', 'cat']:
                model.fit(X_train, y_train, verbose=False)
            else:
                model.fit(X_train, y_train)
        
        return self
    
    def predict(self, X_test):
        """Make ensemble predictions"""
        # Apply feature selection if used
        if self.selected_features is not None:
            X_test = X_test[:, self.selected_features]
        
        # Get base model predictions
        base_predictions = np.zeros((X_test.shape[0], len(self.base_models)))
        
        for idx, (name, model) in enumerate(self.base_models.items()):
            base_predictions[:, idx] = model.predict(X_test)
        
        # Meta model prediction
        final_predictions = self.meta_model.predict(base_predictions)
        
        # Ensure non-negative predictions
        final_predictions = np.maximum(final_predictions, 0)
        
        return final_predictions

# ========================
# MAIN PIPELINE
# ========================

def main():
    """Main execution pipeline"""
    
    try:
        # Feature engineering
        print("\nğŸ”§ FEATURE ENGINEERING")
        print("="*50)
        
        feature_engineer = AdvancedFeatureEngineer()
        train_df_fe = feature_engineer.engineer_features(train_df)
        test_df_fe = feature_engineer.engineer_features(test_df)
        
        # Separate features and target
        target_column = 'Premium Amount'
        y_train = train_df_fe[target_column].values
        X_train_df = train_df_fe.drop(columns=[target_column, 'id'])
        X_test_df = test_df_fe.drop(columns=['id'])
        
        print(f"âœ… Features engineered: {X_train_df.shape[1]} features")
        
        # Preprocessing
        print("\nğŸ”„ PREPROCESSING")
        print("="*50)
        
        preprocessor = MemoryEfficientPreprocessor()
        preprocessor.identify_features(X_train_df, target_column)
        
        X_train_processed = preprocessor.preprocess(X_train_df, y_train, is_train=True)
        X_test_processed = preprocessor.preprocess(X_test_df, is_train=False)
        
        # Ensure same columns
        train_cols = set(X_train_processed.columns)
        test_cols = set(X_test_processed.columns)
        
        # Add missing columns to test
        for col in train_cols - test_cols:
            X_test_processed[col] = 0
        
        # Remove extra columns from test
        X_test_processed = X_test_processed[X_train_processed.columns]
        
        print(f"âœ… Final shape - Train: {X_train_processed.shape}, Test: {X_test_processed.shape}")
        
        # Convert to numpy arrays for modeling
        X_train_final = X_train_processed.values
        X_test_final = X_test_processed.values
        
        # Free memory
        del train_df_fe, test_df_fe, X_train_df, X_test_df, X_train_processed, X_test_processed
        gc.collect()
        
        # Model training
        print("\nğŸš€ MODEL TRAINING")
        print("="*50)
        
        ensemble = AdvancedEnsembleModel()
        ensemble.train_with_cv(X_train_final, y_train, n_folds=5)
        
        # Make predictions
        print("\nğŸ“Š GENERATING PREDICTIONS")
        print("="*50)
        
        predictions = ensemble.predict(X_test_final)
        
        # Post-processing predictions
        predictions = np.round(predictions, 2)
        predictions = np.clip(predictions, 0, 5000)  # Reasonable bounds
        
        print(f"âœ… Predictions generated")
        print(f"   Mean: ${predictions.mean():.2f}")
        print(f"   Std: ${predictions.std():.2f}")
        print(f"   Min: ${predictions.min():.2f}")
        print(f"   Max: ${predictions.max():.2f}")
        
        # Create submission
        submission_df = pd.DataFrame({
            'id': test_ids,
            'Premium Amount': predictions
        })
        
        # Save submission
        submission_df.to_csv('submission_advanced.csv', index=False)
        print("\nâœ… Submission saved to 'submission_advanced.csv'")
        
        # Plot prediction distribution
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.hist(y_train, bins=50, alpha=0.5, label='Training', color='blue')
        plt.hist(predictions, bins=50, alpha=0.5, label='Predictions', color='red')
        plt.xlabel('Premium Amount')
        plt.ylabel('Frequency')
        plt.title('Distribution Comparison')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.scatter(range(len(predictions[:1000])), predictions[:1000], alpha=0.5)
        plt.xlabel('Sample Index')
        plt.ylabel('Predicted Premium')
        plt.title('First 1000 Predictions')
        
        plt.tight_layout()
        plt.savefig('predictions_analysis.png')
        plt.show()
        
        print("\n" + "="*100)
        print("ğŸ�‰ PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"Finished at: {datetime.now()}")
        print("="*100)
        
    except Exception as e:
        print(f"\nâ�Œ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Clean up memory
        gc.collect()

if __name__ == "__main__":
    main()

