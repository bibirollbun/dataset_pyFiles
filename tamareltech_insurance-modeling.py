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


# Install necessary packages with specific versions to avoid conflicts
!pip install numpy pandas scikit-learn==1.2.2 matplotlib seaborn optuna tpot
!pip install flaml h2o_automl lightgbm xgboost catboost
!pip install auto-sklearn==0.15.0

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_log_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgb
import optuna
import time
import logging
import warnings
import traceback
from typing import Dict, List, Tuple, Union, Optional
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("automl_comparison.log")
    ]
)
logger = logging.getLogger(__name__)

class AutoMLComparison:
    """Compare multiple AutoML frameworks for insurance premium prediction"""
    
    def __init__(self, input_path: str, output_path: str = './'):
        """Initialize comparison framework"""
        self.input_path = input_path
        self.output_path = output_path
        self.target_column = 'Premium Amount'
        self.id_column = 'id'
        self.start_time = time.time()
        
        # Create necessary directories
        os.makedirs(self.output_path, exist_ok=True)
        os.makedirs(os.path.join(self.output_path, 'plots'), exist_ok=True)
        os.makedirs(os.path.join(self.output_path, 'models'), exist_ok=True)
        os.makedirs(os.path.join(self.output_path, 'submissions'), exist_ok=True)
        
        # Configuration settings
        self.n_cv_folds = 5
        self.optuna_trials = 100
        self.automl_time_budget = 1800  # 30 minutes per framework
        self.random_seed = 42
        
        # Track performance of all models
        self.framework_status = {}
        self.model_performances = {}
        self.predictions = {}
        
        # Initialize frameworks status
        self.framework_status = {
            "Baseline": "Not Started",
            "Tuned": "Not Started",
            "FLAML": "Not Started",
            "TPOT": "Not Started",
            "H2O AutoML": "Not Started",
            "Auto-Sklearn": "Not Started"
        }
    
    def load_data(self) -> bool:
        """Load and perform initial data analysis"""
        logger.info("Loading datasets...")
        
        try:
            # Load raw data
            self.train = pd.read_csv(os.path.join(self.input_path, 'train.csv'))
            self.test = pd.read_csv(os.path.join(self.input_path, 'test.csv'))
            
            logger.info(f"Train shape: {self.train.shape}, Test shape: {self.test.shape}")
            
            # Extract target variable and test IDs
            self.y_train = self.train[self.target_column].copy()
            self.test_ids = self.test[self.id_column].copy()
            
            # Analyze target distribution
            logger.info(f"Target summary - Mean: {self.y_train.mean():.2f}, "
                        f"Std: {self.y_train.std():.2f}, "
                        f"Min: {self.y_train.min():.2f}, "
                        f"Max: {self.y_train.max():.2f}")
            
            # Identify feature types
            self.categorical_cols = []
            self.numerical_cols = []
            
            for col in self.train.columns:
                if col == self.id_column or col == self.target_column:
                    continue
                    
                if self.train[col].dtype == 'object' or col in [
                    'Gender', 'Marital Status', 'Education Level', 'Occupation', 'Location'
                ]:
                    self.categorical_cols.append(col)
                else:
                    self.numerical_cols.append(col)
            
            logger.info(f"Identified {len(self.categorical_cols)} categorical and {len(self.numerical_cols)} numerical columns")
            return True
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    def engineer_features(self) -> bool:
        """Create domain-specific features for insurance premium prediction"""
        logger.info("Engineering features...")
        
        try:
            # Create working copies
            self.train_processed = self.train.copy()
            self.test_processed = self.test.copy()
            
            # Age-related features
            if 'Age' in self.train.columns:
                # Non-linear age effects
                self.train_processed['Age_Squared'] = self.train_processed['Age'] ** 2
                self.test_processed['Age_Squared'] = self.test_processed['Age'] ** 2
                
                self.train_processed['Log_Age'] = np.log1p(self.train_processed['Age'])
                self.test_processed['Log_Age'] = np.log1p(self.test_processed['Age'])
                
                # Age risk groups
                bins = [0, 25, 35, 45, 55, 65, 100]
                labels = [1, 2, 3, 4, 5, 6]  # Numeric labels for stability
                
                self.train_processed['Age_Group'] = pd.cut(
                    self.train_processed['Age'], 
                    bins=bins, 
                    labels=labels,
                    include_lowest=True
                )
                self.test_processed['Age_Group'] = pd.cut(
                    self.test_processed['Age'], 
                    bins=bins, 
                    labels=labels,
                    include_lowest=True
                )
                
                # Handle potential NaNs
                self.train_processed['Age_Group'] = self.train_processed['Age_Group'].fillna(3)
                self.test_processed['Age_Group'] = self.test_processed['Age_Group'].fillna(3)
            
            # Income-related features
            if 'Annual Income' in self.train.columns:
                # Log transform for skewed data
                self.train_processed['Log_Income'] = np.log1p(self.train_processed['Annual Income'])
                self.test_processed['Log_Income'] = np.log1p(self.test_processed['Annual Income'])
                
                # Income brackets
                income_bins = [0, 20000, 40000, 60000, 100000, float('inf')]
                income_labels = [1, 2, 3, 4, 5]
                
                self.train_processed['Income_Bracket'] = pd.cut(
                    self.train_processed['Annual Income'],
                    bins=income_bins,
                    labels=income_labels,
                    include_lowest=True
                )
                self.test_processed['Income_Bracket'] = pd.cut(
                    self.test_processed['Annual Income'],
                    bins=income_bins,
                    labels=income_labels,
                    include_lowest=True
                )
                
                # Handle potential NaNs
                self.train_processed['Income_Bracket'] = self.train_processed['Income_Bracket'].fillna(3)
                self.test_processed['Income_Bracket'] = self.test_processed['Income_Bracket'].fillna(3)
            
            # Health-related features
            if 'Health Score' in self.train.columns:
                # Inverse health score (higher = higher risk)
                max_health = self.train_processed['Health Score'].max()
                self.train_processed['Health_Risk'] = max_health - self.train_processed['Health Score']
                self.test_processed['Health_Risk'] = max_health - self.test_processed['Health Score']
            
            # Family size
            if 'Number of Dependents' in self.train.columns:
                self.train_processed['Family_Size'] = self.train_processed['Number of Dependents'].fillna(0) + 1
                self.test_processed['Family_Size'] = self.test_processed['Number of Dependents'].fillna(0) + 1
                
                # Income per person
                if 'Annual Income' in self.train.columns:
                    self.train_processed['Income_Per_Person'] = (
                        self.train_processed['Annual Income'] / self.train_processed['Family_Size'].replace(0, 1)
                    )
                    self.test_processed['Income_Per_Person'] = (
                        self.test_processed['Annual Income'] / self.test_processed['Family_Size'].replace(0, 1)
                    )
            
            # Interaction features
            if all(col in self.train.columns for col in ['Age', 'Health Score']):
                self.train_processed['Age_Health'] = self.train_processed['Age'] * self.train_processed['Health Score']
                self.test_processed['Age_Health'] = self.test_processed['Age'] * self.test_processed['Health Score']
            
            logger.info(f"Feature engineering complete. New feature count: {self.train_processed.shape[1]}")
            return True
            
        except Exception as e:
            logger.error(f"Error in feature engineering: {str(e)}")
            logger.error(traceback.format_exc())
            # Fall back to original data
            self.train_processed = self.train.copy()
            self.test_processed = self.test.copy()
            logger.info("Falling back to original features")
            return True  # Continue pipeline despite errors
    
    def preprocess_data(self) -> bool:
        """Prepare data for modeling"""
        logger.info("Preprocessing data...")
        
        try:
            # Create working copies
            train_df = self.train_processed.copy()
            test_df = self.test_processed.copy()
            
            # Remove ID and target
            if self.id_column in train_df.columns:
                train_df = train_df.drop(self.id_column, axis=1)
            if self.target_column in train_df.columns:
                train_df = train_df.drop(self.target_column, axis=1)
            if self.id_column in test_df.columns:
                test_df = test_df.drop(self.id_column, axis=1)
            
            # Update categorical columns list
            categorical_cols = self.categorical_cols.copy()
            for col in train_df.columns:
                if col.endswith('_Group') or col.endswith('_Bracket'):
                    if col not in categorical_cols:
                        categorical_cols.append(col)
            
            # Handle missing values and encode categorical features
            for col in train_df.columns:
                if col in categorical_cols:
                    # Fill missing values
                    train_df[col] = train_df[col].fillna('Unknown')
                    test_df[col] = test_df[col].fillna('Unknown')
                    
                    # Label encoding
                    le = LabelEncoder()
                    combined = pd.concat([train_df[col], test_df[col]]).astype(str)
                    le.fit(combined)
                    train_df[col] = le.transform(train_df[col].astype(str))
                    test_df[col] = test_df[col].astype(str)
                    test_df[col] = le.transform(test_df[col])
                else:
                    # Handle numerical missing values
                    median_val = train_df[col].median()
                    train_df[col] = train_df[col].fillna(median_val)
                    test_df[col] = test_df[col].fillna(median_val)
            
            # Final check for missing values
            if train_df.isnull().sum().sum() > 0 or test_df.isnull().sum().sum() > 0:
                train_df = train_df.fillna(0)
                test_df = test_df.fillna(0)
            
            # Store processed data
            self.X_train = train_df
            self.X_test = test_df
            
            logger.info(f"Preprocessing complete. Final data shapes - Train: {self.X_train.shape}, Test: {self.X_test.shape}")
            return True
            
        except Exception as e:
            logger.error(f"Error in preprocessing: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    def define_rmsle_scorer(self):
        """Define custom RMSLE scorer for model evaluation"""
        # RMSLE calculation function
        def rmsle(y_true, y_pred):
            y_true = np.maximum(y_true, 0.001)
            y_pred = np.maximum(y_pred, 0.001)
            return np.sqrt(mean_squared_log_error(y_true, y_pred))
        
        # Scorer function for cross-validation
        def rmsle_scorer(estimator, X, y):
            y_pred = estimator.predict(X)
            return -rmsle(y, y_pred)  # Negative because sklearn maximizes scores
        
        return rmsle, rmsle_scorer
    
    def train_baseline_models(self) -> bool:
        """Train models with default parameters (no tuning)"""
        logger.info("Training baseline models with default parameters...")
        self.framework_status["Baseline"] = "Running"
        
        try:
            # Define models to evaluate
            baseline_models = {
                'Random Forest': RandomForestRegressor(random_state=self.random_seed),
                'Gradient Boosting': GradientBoostingRegressor(random_state=self.random_seed),
                'XGBoost': xgb.XGBRegressor(objective='reg:squarederror', random_state=self.random_seed),
                'LightGBM': lgb.LGBMRegressor(objective='regression', random_state=self.random_seed)
            }
            
            # Define evaluation metric
            rmsle_func, rmsle_scorer = self.define_rmsle_scorer()
            
            # Evaluate each model using cross-validation
            baseline_scores = {}
            baseline_times = {}
            
            for name, model in baseline_models.items():
                logger.info(f"Evaluating baseline {name}...")
                start_time = time.time()
                
                # Perform cross-validation
                scores = cross_val_score(
                    model, 
                    self.X_train, 
                    self.y_train, 
                    cv=self.n_cv_folds, 
                    scoring=rmsle_scorer,
                    n_jobs=-1
                )
                
                # Convert scores to positive RMSLE
                rmsle_scores = -scores
                
                # Store results
                baseline_scores[name] = rmsle_scores.mean()
                baseline_times[name] = time.time() - start_time
                
                logger.info(f"  {name} - Mean RMSLE: {rmsle_scores.mean():.5f}, Time: {baseline_times[name]:.2f}s")
                
                # Store in overall performance tracker
                self.model_performances[f"Baseline {name}"] = {
                    'Framework': 'Baseline',
                    'RMSLE': rmsle_scores.mean(),
                    'Std': rmsle_scores.std(),
                    'Training Time': baseline_times[name]
                }
            
            # Train best baseline model on full data
            best_model_name = min(baseline_scores, key=baseline_scores.get)
            logger.info(f"Best baseline model: {best_model_name} (RMSLE: {baseline_scores[best_model_name]:.5f})")
            
            best_baseline = baseline_models[best_model_name]
            best_baseline.fit(self.X_train, self.y_train)
            
            # Make predictions with best baseline model
            self.predictions['Baseline'] = best_baseline.predict(self.X_test)
            self.best_baseline_model = best_baseline
            self.best_baseline_name = best_model_name
            
            self.framework_status["Baseline"] = "Completed"
            return True
            
        except Exception as e:
            logger.error(f"Error in baseline model training: {str(e)}")
            logger.error(traceback.format_exc())
            self.framework_status["Baseline"] = f"Failed: {str(e)}"
            return False
    
    def optimize_hyperparameters(self, model_type: str) -> Tuple[Dict, float]:
        """Optimize hyperparameters using Optuna for a given model type"""
        logger.info(f"Optimizing hyperparameters for {model_type}...")
        
        # Define evaluation metric
        rmsle_func, _ = self.define_rmsle_scorer()
        
        # Create objective function for Optuna
        def objective(trial):
            if model_type == 'Random Forest':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                    'max_depth': trial.suggest_int('max_depth', 5, 30),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                    'max_features': trial.suggest_float('max_features', 0.5, 1.0),
                    'random_state': self.random_seed
                }
                model = RandomForestRegressor(**params)
                
            elif model_type == 'Gradient Boosting':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                    'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                    'random_state': self.random_seed
                }
                model = GradientBoostingRegressor(**params)
                
            elif model_type == 'XGBoost':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                    'random_state': self.random_seed
                }
                model = xgb.XGBRegressor(objective='reg:squarederror', **params)
                
            elif model_type == 'LightGBM':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'num_leaves': trial.suggest_int('num_leaves', 31, 255),
                    'max_depth': trial.suggest_int('max_depth', 3, 12),
                    'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                    'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                    'random_state': self.random_seed
                }
                model = lgb.LGBMRegressor(objective='regression', **params)
                
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Evaluate model using cross-validation
            kf = KFold(n_splits=self.n_cv_folds, shuffle=True, random_state=self.random_seed)
            scores = []
            
            for train_idx, val_idx in kf.split(self.X_train):
                X_train_fold, X_val_fold = self.X_train.iloc[train_idx], self.X_train.iloc[val_idx]
                y_train_fold, y_val_fold = self.y_train.iloc[train_idx], self.y_train.iloc[val_idx]
                
                # Train model
                model.fit(X_train_fold, y_train_fold)
                
                # Evaluate
                y_pred = model.predict(X_val_fold)
                fold_rmsle = rmsle_func(y_val_fold, y_pred)
                scores.append(fold_rmsle)
            
            # Return mean RMSLE score
            return np.mean(scores)
        
        # Create Optuna study
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=self.optuna_trials, timeout=1200)  # 20 min timeout
        
        # Get best parameters
        best_params = study.best_params
        best_score = study.best_value
        
        logger.info(f"Best {model_type} hyperparameters: {best_params}")
        logger.info(f"Best {model_type} RMSLE: {best_score:.5f}")
        
        return best_params, best_score
    
    def train_tuned_models(self) -> bool:
        """Train models with optimized hyperparameters"""
        logger.info("Training models with optimized hyperparameters...")
        self.framework_status["Tuned"] = "Running"
        
        try:
            # Choose top 2 baseline models to optimize
            baseline_performances = {name: self.model_performances[name]['RMSLE'] 
                                  for name in self.model_performances 
                                  if name.startswith('Baseline')}
            
            sorted_baselines = sorted(baseline_performances.items(), key=lambda x: x[1])
            top_models = [name.split(' ', 1)[1] for name, _ in sorted_baselines[:2]]
            
            logger.info(f"Optimizing top 2 models: {top_models}")
            
            # Optimize and train each model
            tuned_models = {}
            tuned_scores = {}
            tuned_times = {}
            
            for model_name in top_models:
                start_time = time.time()
                
                # Optimize hyperparameters
                best_params, best_cv_score = self.optimize_hyperparameters(model_name)
                
                # Initialize model with best parameters
                if model_name == 'Random Forest':
                    model = RandomForestRegressor(random_state=self.random_seed, **best_params)
                elif model_name == 'Gradient Boosting':
                    model = GradientBoostingRegressor(random_state=self.random_seed, **best_params)
                elif model_name == 'XGBoost':
                    model = xgb.XGBRegressor(objective='reg:squarederror', random_state=self.random_seed, **best_params)
                elif model_name == 'LightGBM':
                    model = lgb.LGBMRegressor(objective='regression', random_state=self.random_seed, **best_params)
                
                # Train model on full data
                model.fit(self.X_train, self.y_train)
                
                # Store model and results
                tuned_models[model_name] = model
                tuned_scores[model_name] = best_cv_score
                tuned_times[model_name] = time.time() - start_time
                
                # Store in overall performance tracker
                self.model_performances[f"Tuned {model_name}"] = {
                    'Framework': 'Tuned',
                    'RMSLE': best_cv_score,
                    'Std': 0,  # We don't have std from Optuna
                    'Training Time': tuned_times[model_name],
                    'Parameters': best_params
                }
                
                logger.info(f"  {model_name} - Tuned RMSLE: {best_cv_score:.5f}, Time: {tuned_times[model_name]:.2f}s")
            
            # Choose best tuned model
            best_model_name = min(tuned_scores, key=tuned_scores.get)
            logger.info(f"Best tuned model: {best_model_name} (RMSLE: {tuned_scores[best_model_name]:.5f})")
            
            best_tuned = tuned_models[best_model_name]
            
            # Make predictions with best tuned model
            self.predictions['Tuned'] = best_tuned.predict(self.X_test)
            self.best_tuned_model = best_tuned
            self.best_tuned_name = best_model_name
            
            self.framework_status["Tuned"] = "Completed"
            return True
            
        except Exception as e:
            logger.error(f"Error in tuned model training: {str(e)}")
            logger.error(traceback.format_exc())
            self.framework_status["Tuned"] = f"Failed: {str(e)}"
            return False
    
    def train_flaml(self) -> bool:
        """Train models using FLAML AutoML"""
        logger.info("Training models with FLAML AutoML...")
        self.framework_status["FLAML"] = "Running"
        
        try:
            from flaml import AutoML
            
            # Define RMSLE for evaluation
            rmsle_func, _ = self.define_rmsle_scorer()
            
            # Import custom metric to FLAML
            from flaml.ml import make_scorer
            custom_rmsle_metric = make_scorer(
                'custom_rmsle',
                lambda y, y_pred: -rmsle_func(y, y_pred),  # Negative because FLAML maximizes metrics
                greater_is_better=False
            )
            
            # Initialize FLAML
            automl = AutoML()
            
            # Configure settings
            settings = {
                "time_budget": self.automl_time_budget,
                "metric": custom_rmsle_metric,
                "task": "regression",
                "estimator_list": ["lgbm", "xgboost", "rf", "catboost", "extra_trees"],
                "early_stop": True,
                "n_jobs": -1,
                "verbose": 1
            }
            
            # Train FLAML
            start_time = time.time()
            automl.fit(X_train=self.X_train, y_train=self.y_train, **settings)
            training_time = time.time() - start_time
            
            # Get model performance
            best_estimator = automl.best_estimator
            best_loss = automl.best_loss  # This is already the negative RMSLE
            
            logger.info(f"FLAML best model: {best_estimator}")
            logger.info(f"FLAML best RMSLE: {-best_loss:.5f}")
            
            # Make predictions
            self.predictions['FLAML'] = automl.predict(self.X_test)
            
            # Store model
            self.flaml_model = automl
            
            # Store performance metrics
            self.model_performances["FLAML"] = {
                'Framework': 'FLAML',
                'RMSLE': -best_loss,
                'Std': 0,  # FLAML doesn't provide std
                'Training Time': training_time,
                'Best Model': best_estimator
            }
            
            self.framework_status["FLAML"] = "Completed"
            return True
            
        except Exception as e:
            logger.error(f"Error in FLAML training: {str(e)}")
            logger.error(traceback.format_exc())
            self.framework_status["FLAML"] = f"Failed: {str(e)}"
            return False
    
    def train_tpot(self) -> bool:
        """Train models using TPOT AutoML"""
        logger.info("Training models with TPOT AutoML...")
        self.framework_status["TPOT"] = "Running"
        
        try:
            from tpot import TPOTRegressor
            from sklearn.metrics import make_scorer
            
            # Define RMSLE for evaluation
            rmsle_func, _ = self.define_rmsle_scorer()
            rmsle_scorer = make_scorer(rmsle_func, greater_is_better=False)
            
            # Initialize TPOT
            tpot = TPOTRegressor(
                generations=5,
                population_size=20,
                verbosity=2,
                scoring=rmsle_scorer,
                cv=5,
                random_state=self.random_seed,
                n_jobs=-1,
                max_time_mins=self.automl_time_budget/60,  # Convert to minutes
                max_eval_time_mins=5
            )
            
            # Train TPOT
            start_time = time.time()
            tpot.fit(self.X_train, self.y_train)
            training_time = time.time() - start_time
            
            # Evaluate performance
            # TPOT doesn't directly provide CV scores, use the best pipeline
            best_pipeline = tpot.fitted_pipeline_
            
            # Perform CV to get RMSLE
            _, rmsle_scorer = self.define_rmsle_scorer()
            scores = cross_val_score(
                best_pipeline, 
                self.X_train, 
                self.y_train, 
                cv=self.n_cv_folds, 
                scoring=rmsle_scorer,
                n_jobs=-1
            )
            
            # Convert scores to positive RMSLE
            rmsle_scores = -scores
            
            logger.info(f"TPOT best pipeline RMSLE: {rmsle_scores.mean():.5f}")
            
            # Make predictions
            self.predictions['TPOT'] = tpot.predict(self.X_test)
            
            # Store model
            self.tpot_model = tpot
            
            # Store performance metrics
            self.model_performances["TPOT"] = {
                'Framework': 'TPOT',
                'RMSLE': rmsle_scores.mean(),
                'Std': rmsle_scores.std(),
                'Training Time': training_time,
                'Best Pipeline': str(best_pipeline)
            }
            
            self.framework_status["TPOT"] = "Completed"
            return True
            
        except Exception as e:
            logger.error(f"Error in TPOT training: {str(e)}")
            logger.error(traceback.format_exc())
            self.framework_status["TPOT"] = f"Failed: {str(e)}"
            return False
    
    def train_h2o_automl(self) -> bool:
        """Train models using H2O AutoML"""
        logger.info("Training models with H2O AutoML...")
        self.framework_status["H2O AutoML"] = "Running"
        
        try:
            import h2o
            from h2o.automl import H2OAutoML
            
            # Initialize H2O
            h2o.init()
            
            # Convert data to H2O frames
            train_frame = h2o.H2OFrame(pd.concat([self.X_train, self.y_train], axis=1))
            test_frame = h2o.H2OFrame(self.X_test)
            
            # Identify features and target
            features = list(self.X_train.columns)
            target = self.target_column
            
            # Initialize H2O AutoML
            aml = H2OAutoML(
                max_runtime_secs=self.automl_time_budget,
                seed=self.random_seed,
                max_models=20,
                nfolds=5,
                verbosity="info",
                keep_cross_validation_predictions=True
            )
            
            # Train models
            start_time = time.time()
            aml.train(x=features, y=target, training_frame=train_frame)
            training_time = time.time() - start_time
            
            # Get leaderboard
            lb = aml.leaderboard
            top_model_id = lb[0, 'model_id']
            best_model = h2o.get_model(top_model_id)
            
            # Evaluate performance
            best_score = aml.leader.model_performance(xval=True).rmse()  # RMSE, not RMSLE
            
            # H2O doesn't provide direct RMSLE, we'll convert predictions to estimate it
            # Make predictions
            predictions_h2o = aml.predict(test_frame)
            predictions_array = h2o.as_list(predictions_h2o)['predict'].values
            
            self.predictions['H2O AutoML'] = predictions_array
            
            # Store model info
            self.h2o_automl = aml
            
            # Store performance metrics (note: not directly RMSLE)
            self.model_performances["H2O AutoML"] = {
                'Framework': 'H2O AutoML',
                'RMSLE': best_score,  # Approximation
                'Std': 0,  # H2O doesn't directly provide std
                'Training Time': training_time,
                'Best Model': top_model_id
            }
            
            self.framework_status["H2O AutoML"] = "Completed"
            
            # Shutdown H2O
            h2o.cluster().shutdown()
            
            return True
            
        except Exception as e:
            logger.error(f"Error in H2O AutoML training: {str(e)}")
            logger.error(traceback.format_exc())
            self.framework_status["H2O AutoML"] = f"Failed: {str(e)}"
            try:
                h2o.cluster().shutdown()
            except:
                pass
            return False
    
    def train_autosklearn(self) -> bool:
        """Train models using Auto-Sklearn"""
        logger.info("Training models with Auto-Sklearn...")
        self.framework_status["Auto-Sklearn"] = "Running"
        
        try:
            import autosklearn.regression
            
            # Define RMSLE for evaluation
            rmsle_func, _ = self.define_rmsle_scorer()
            
            # Initialize Auto-Sklearn
            autosk = autosklearn.regression.AutoSklearnRegressor(
                time_left_for_this_task=self.automl_time_budget,
                per_run_time_limit=300,  # 5 minutes per run
                n_jobs=-1,
                ensemble_size=50,
                max_models_on_disc=100,
                resampling_strategy='cv',
                resampling_strategy_arguments={'folds': 5},
                seed=self.random_seed
            )
            
            # Train model
            start_time = time.time()
            autosk.fit(self.X_train.copy(), self.y_train.copy())
            training_time = time.time() - start_time
            
            # Get performance metrics
            # Auto-sklearn doesn't provide RMSLE directly, need to convert
            cv_results = autosk.cv_results_
            best_score = -cv_results['mean_test_score'][0]  # Mean negative MSE
            
            # Make predictions
            predictions_autosk = autosk.predict(self.X_test)
            self.predictions['Auto-Sklearn'] = predictions_autosk
            
            # Get models in ensemble
            models_info = autosk.show_models()
            
            # Store model
            self.autosklearn_model = autosk
            
            # Store performance metrics (note: not directly RMSLE)
            self.model_performances["Auto-Sklearn"] = {
                'Framework': 'Auto-Sklearn',
                'RMSLE': np.sqrt(best_score),  # Approximation, converting MSE to RMSE
                'Std': np.std(cv_results['mean_test_score']),
                'Training Time': training_time,
                'Models Info': str(models_info)
            }
            
            self.framework_status["Auto-Sklearn"] = "Completed"
            return True
            
        except Exception as e:
            logger.error(f"Error in Auto-Sklearn training: {str(e)}")
            logger.error(traceback.format_exc())
            self.framework_status["Auto-Sklearn"] = f"Failed: {str(e)}"
            return False
    
    def blend_predictions(self) -> bool:
        """Create a blended model combining predictions from all frameworks"""
        logger.info("Creating blended prediction...")
        
        try:
            # Get successful frameworks
            successful_frameworks = [
                framework for framework, status in self.framework_status.items()
                if status == "Completed" and framework in self.predictions
            ]
            
            if len(successful_frameworks) <= 1:
                logger.warning("Not enough successful frameworks for blending")
                # Use the only successful framework or fallback to Baseline
                for framework in successful_frameworks:
                    self.final_predictions = self.predictions[framework]
                    logger.info(f"Using predictions from {framework} as final")
                    return True
                
                # If no successful frameworks, use Baseline if available
                if 'Baseline' in self.predictions:
                    self.final_predictions = self.predictions['Baseline']
                    logger.info("Using Baseline predictions as final")
                    return True
                
                # Last resort: mean of target
                self.final_predictions = np.full(len(self.X_test), self.y_train.mean())
                logger.info("Using mean target value as final prediction")
                return True
            
            # Calculate weights based on RMSLE (if available)
            weights = {}
            for framework in successful_frameworks:
                if framework in self.model_performances:
                    rmsle = self.model_performances[framework].get('RMSLE')
                    if rmsle is not None and rmsle > 0:
                        weights[framework] = 1 / rmsle
                    else:
                        weights[framework] = 1
                else:
                    weights[framework] = 1
            
            # Normalize weights
            total_weight = sum(weights.values())
            for framework in weights:
                weights[framework] /= total_weight
            
            logger.info(f"Blend weights: {weights}")
            
            # Create blended prediction
            self.final_predictions = np.zeros(len(self.X_test))
            for framework, weight in weights.items():
                self.final_predictions += weight * self.predictions[framework]
            
            return True
            
        except Exception as e:
            logger.error(f"Error in prediction blending: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Fall back to best individual framework
            best_framework = None
            best_score = float('inf')
            
            for framework, metrics in self.model_performances.items():
                if framework in self.predictions:
                    rmsle = metrics.get('RMSLE')
                    if rmsle is not None and rmsle < best_score:
                        best_score = rmsle
                        best_framework = framework
            
            if best_framework is not None:
                self.final_predictions = self.predictions[best_framework]
                logger.info(f"Falling back to predictions from {best_framework}")
            else:
                # Last resort: mean of target
                self.final_predictions = np.full(len(self.X_test), self.y_train.mean())
                logger.info("Using mean target value as final prediction")
            
            return True
    
    def create_submissions(self) -> bool:
        """Create submission files for each framework and the blended model"""
        logger.info("Creating submission files...")
        
        try:
            # Create directory for individual framework submissions
            frameworks_dir = os.path.join(self.output_path, 'submissions', 'frameworks')
            os.makedirs(frameworks_dir, exist_ok=True)
            
            # Create individual submissions for each framework
            for framework, predictions in self.predictions.items():
                # Ensure predictions are positive (required for RMSLE)
                framework_preds = np.maximum(predictions, 0.001)
                
                # Create submission dataframe
                submission = pd.DataFrame({
                    self.id_column: self.test_ids,
                    self.target_column: framework_preds
                })
                
                # Save to CSV
                framework_path = os.path.join(frameworks_dir, f'{framework.replace(" ", "_")}_submission.csv')
                submission.to_csv(framework_path, index=False)
                logger.info(f"{framework} submission saved to {framework_path}")
            
            # Create blended submission
            blended_preds = np.maximum(self.final_predictions, 0.001)
            blended_submission = pd.DataFrame({
                self.id_column: self.test_ids,
                self.target_column: blended_preds
            })
            
            # Save blended submission as main submission
            main_path = os.path.join(self.output_path, 'submissions', 'submission.csv')
            blended_submission.to_csv(main_path, index=False)
            logger.info(f"Main (blended) submission saved to {main_path}")
            
            # Also save a copy in root output directory for easier access
            root_path = os.path.join(self.output_path, 'submission.csv')
            blended_submission.to_csv(root_path, index=False)
            
            return True
            
        except Exception as e:
            logger.error(f"Error creating submissions: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Try to create a minimal submission as fallback
            try:
                fallback_preds = np.full(len(self.test_ids), self.y_train.mean())
                fallback_submission = pd.DataFrame({
                    self.id_column: self.test_ids,
                    self.target_column: fallback_preds
                })
                
                fallback_path = os.path.join(self.output_path, 'submission.csv')
                fallback_submission.to_csv(fallback_path, index=False)
                logger.info(f"Fallback submission saved to {fallback_path}")
                
                return True
            except Exception as fallback_error:
                logger.error(f"Failed to create fallback submission: {str(fallback_error)}")
                return False
    
    def visualize_results(self) -> bool:
        """Create visualizations comparing all frameworks"""
        logger.info("Creating performance comparison visualizations...")
        
        try:
            # Extract performance metrics for visualization
            models = []
            rmsle_scores = []
            training_times = []
            frameworks = []
            
            for name, metrics in self.model_performances.items():
                models.append(name)
                rmsle_scores.append(metrics.get('RMSLE', float('nan')))
                training_times.append(metrics.get('Training Time', float('nan')))
                frameworks.append(metrics.get('Framework', 'Unknown'))
            
            # Create performance dataframe
            performance_df = pd.DataFrame({
                'Model': models,
                'RMSLE': rmsle_scores,
                'Training Time': training_times,
                'Framework': frameworks
            })
            
            # Remove rows with NaN RMSLE
            performance_df = performance_df.dropna(subset=['RMSLE'])
            
            # Sort by RMSLE for better visualization
            performance_df_sorted = performance_df.sort_values('RMSLE')
            
            # Plot RMSLE comparison
            plt.figure(figsize=(12, 8))
            sns.barplot(x='RMSLE', y='Model', hue='Framework', data=performance_df_sorted)
            plt.title('Model Performance Comparison (RMSLE, lower is better)')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_path, 'plots', 'rmsle_comparison.png'))
            
            # Plot training time comparison
            plt.figure(figsize=(12, 8))
            sns.barplot(x='Training Time', y='Model', hue='Framework', data=performance_df_sorted)
            plt.title('Model Training Time Comparison (seconds)')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_path, 'plots', 'time_comparison.png'))
            
            # Plot prediction distributions for successful frameworks
            plt.figure(figsize=(12, 8))
            for framework, predictions in self.predictions.items():
                sns.kdeplot(predictions, label=framework, alpha=0.7)
            
            plt.title('Prediction Distributions by Framework')
            plt.xlabel('Predicted Premium Amount')
            plt.ylabel('Density')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_path, 'plots', 'prediction_distributions.png'))
            
            # Create and save performance summary
            performance_df_sorted.to_csv(os.path.join(self.output_path, 'model_performance_summary.csv'), index=False)
            
            # Calculate and save framework status summary
            status_df = pd.DataFrame({
                'Framework': list(self.framework_status.keys()),
                'Status': list(self.framework_status.values())
            })
            status_df.to_csv(os.path.join(self.output_path, 'framework_status_summary.csv'), index=False)
            
            # Output framework status to console
            logger.info("\nFramework Status Summary:")
            for framework, status in self.framework_status.items():
                logger.info(f"  {framework}: {status}")
            
            # Find best individual framework
            if not performance_df_sorted.empty:
                best_model = performance_df_sorted.iloc[0]['Model']
                best_rmsle = performance_df_sorted.iloc[0]['RMSLE']
                best_framework = performance_df_sorted.iloc[0]['Framework']
                
                logger.info(f"\nBest model: {best_model} ({best_framework}) with RMSLE: {best_rmsle:.5f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    def run_pipeline(self) -> bool:
        """Execute the full AutoML comparison pipeline"""
        logger.info("Starting comprehensive AutoML framework comparison for insurance premium prediction...")
        
        # Define pipeline steps with flexible execution
        base_steps = [
            (self.load_data, "Data Loading", True),  # Required
            (self.engineer_features, "Feature Engineering", True),  # Required
            (self.preprocess_data, "Data Preprocessing", True),  # Required
        ]
        
        # AutoML framework steps - each can fail independently
        framework_steps = [
            (self.train_baseline_models, "Baseline Model Training", False),
            (self.train_tuned_models, "Hyperparameter-Tuned Model Training", False),
            (self.train_flaml, "FLAML AutoML Training", False),
            (self.train_tpot, "TPOT AutoML Training", False),
            (self.train_h2o_automl, "H2O AutoML Training", False),
            (self.train_autosklearn, "Auto-Sklearn Training", False),
        ]
        
        # Final steps
        final_steps = [
            (self.blend_predictions, "Model Blending", True),  # Required
            (self.create_submissions, "Submission Creation", True),  # Required
            (self.visualize_results, "Results Visualization", False),  # Optional
        ]
        
        # Execute base steps (required)
        for step_func, step_name, required in base_steps:
            step_start = time.time()
            logger.info(f"Starting: {step_name}")
            
            success = step_func()
            step_time = time.time() - step_start
            
            if not success and required:
                logger.error(f"Pipeline failed at required step: {step_name}")
                return False
            
            logger.info(f"Completed: {step_name} in {step_time:.2f} seconds")
        
        # Execute framework steps (each optional)
        for step_func, step_name, required in framework_steps:
            step_start = time.time()
            logger.info(f"Starting: {step_name}")
            
            success = step_func()  # We continue even if individual frameworks fail
            step_time = time.time() - step_start
            
            logger.info(f"Completed: {step_name} in {step_time:.2f} seconds - Success: {success}")
        
        # Execute final steps
        for step_func, step_name, required in final_steps:
            step_start = time.time()
            logger.info(f"Starting: {step_name}")
            
            success = step_func()
            step_time = time.time() - step_start
            
            if not success and required:
                logger.error(f"Pipeline failed at required step: {step_name}")
                return False
            
            logger.info(f"Completed: {step_name} in {step_time:.2f} seconds")
        
        # Calculate total runtime
        total_time = (time.time() - self.start_time) / 60
        logger.info(f"Pipeline completed in {total_time:.2f} minutes")
        
        return True


# Execute pipeline
if __name__ == "__main__":
    input_path = '/kaggle/input/premiumpulse-risk-modeling/'
    output_path = '/kaggle/working/'
    
    comparison = AutoMLComparison(input_path, output_path)
    comparison.run_pipeline()

