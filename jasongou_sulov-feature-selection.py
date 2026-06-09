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


import pandas as pd
import numpy as np
import json
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from scipy.stats import pearsonr
import time
import warnings
import gc
import os
from datetime import datetime

warnings.filterwarnings('ignore')


class CryptoMarketPredictor:
    """
    A comprehensive framework for training crypto market prediction models
    using different feature selection strategies and generating competition submissions.
    """
    
    def __init__(self, train_path, test_path, sulov_results_path='sulov_selection_results.json'):
        self.train_path = train_path
        self.test_path = test_path
        self.sulov_results_path = sulov_results_path
        self.baseline_features = ["bid_qty", "ask_qty", "buy_qty", "sell_qty", "volume"]
        self.models = {}
        self.feature_sets = {}
        self.predictions = {}
        self.performance_metrics = {}
        
    def load_sulov_results(self):
        """Load and parse SULOV feature selection results from JSON file."""
        print("Loading SULOV feature selection results...")
        
        try:
            with open(self.sulov_results_path, 'r') as f:
                sulov_results = json.load(f)
        except FileNotFoundError:
            print(f"Warning: {self.sulov_results_path} not found. Using default features.")
            return [], []
        
        selected_features = sulov_results.get('selected_features', [])
        cluster_summary = sulov_results.get('cluster_summary', {})
        
        # Extract features with their target correlations
        feature_correlations = []
        for cluster_info in cluster_summary.values():
            feature = cluster_info['representative']
            correlation = cluster_info['target_correlation']
            if feature not in self.baseline_features:
                feature_correlations.append((feature, correlation))
        
        # Sort by correlation and take top 30
        feature_correlations.sort(key=lambda x: x[1], reverse=True)
        top_30_sulov = [feature for feature, _ in feature_correlations[:30]]
        
        print(f"Loaded {len(selected_features)} total SULOV features")
        print(f"Selected top 30 features by target correlation")
        
        return top_30_sulov, feature_correlations
    
    def select_random_features(self, all_features, n_features=30, seed=42):
        """Select random features from the anonymized feature set."""
        np.random.seed(seed)
        
        # Filter out baseline features and any non-feature columns
        anonymized_features = [f for f in all_features 
                             if f not in self.baseline_features 
                             and f not in ['timestamp', 'label']]
        
        # Ensure we have enough features to select from
        if len(anonymized_features) < n_features:
            print(f"Warning: Only {len(anonymized_features)} features available, selecting all")
            return anonymized_features
        
        # Randomly select features
        random_features = np.random.choice(anonymized_features, size=n_features, replace=False)
        
        print(f"Selected {n_features} random features from {len(anonymized_features)} available")
        
        return list(random_features)
    
    def load_and_prepare_data(self, feature_list, sample_size=None):
        """
        Load training and test data with specified features.
        
        Parameters:
        -----------
        feature_list : list
            List of features to use
        sample_size : int or None
            If specified, sample this many rows from training data
        
        Returns:
        --------
        tuple
            (X_train, y_train, X_test, test_ids)
        """
        print(f"\nLoading data with {len(feature_list)} features...")
        
        # Load training data
        print("Loading training data...")
        train_df = pd.read_parquet(self.train_path)
        
        # Apply sampling if requested
        if sample_size and sample_size < len(train_df):
            n_rows = len(train_df)
            
            # Sample more heavily from recent data
            weights = np.linspace(0.5, 1.0, n_rows)
            weights = weights / weights.sum()
            
            sample_indices = np.random.choice(n_rows, size=sample_size, replace=False, p=weights)
            train_df = train_df.iloc[sample_indices].reset_index(drop=True)
            print(f"Sampled {sample_size} rows from {n_rows} total rows")
        
        # Load test data
        print("Loading test data...")
        test_df = pd.read_parquet(self.test_path)
        
        # Verify all requested features exist
        available_features = [f for f in feature_list if f in train_df.columns]
        missing_features = [f for f in feature_list if f not in train_df.columns]
        
        if missing_features:
            print(f"Warning: {len(missing_features)} features not found in data: {missing_features[:5]}...")
            feature_list = available_features
        
        # Extract features and target
        X_train = train_df[feature_list].copy()
        y_train = train_df['label'].copy()
        X_test = test_df[feature_list].copy()
        test_ids = test_df.index
        
        print(f"Training data shape: {X_train.shape}")
        print(f"Test data shape: {X_test.shape}")
        
        # Handle any remaining missing values
        for col in X_train.columns:
            if X_train[col].isna().any():
                median_val = X_train[col].median()
                X_train[col] = X_train[col].fillna(median_val)
                X_test[col] = X_test[col].fillna(median_val)
        
        # Clean up memory
        del train_df, test_df
        gc.collect()
        
        return X_train, y_train, X_test, test_ids
    
    def create_model_configurations(self):
        """Define model configurations for different approaches."""
        return {
            'xgboost_conservative': {
                'model_class': XGBRegressor,
                'params': {
                    'n_estimators': 300,
                    'max_depth': 5,
                    'learning_rate': 0.01,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'gamma': 1,
                    'reg_alpha': 0.1,
                    'reg_lambda': 1,
                    'random_state': 42,
                    'n_jobs': -1,
                    'tree_method': 'hist',
                    'objective': 'reg:squarederror'
                }
            },
            'xgboost_aggressive': {
                'model_class': XGBRegressor,
                'params': {
                    'n_estimators': 500,
                    'max_depth': 8,
                    'learning_rate': 0.03,
                    'subsample': 0.7,
                    'colsample_bytree': 0.7,
                    'gamma': 0.1,
                    'reg_alpha': 0.01,
                    'reg_lambda': 0.1,
                    'random_state': 42,
                    'n_jobs': -1,
                    'tree_method': 'hist',
                    'objective': 'reg:squarederror'
                }
            },
            'lightgbm': {
                'model_class': LGBMRegressor,
                'params': {
                    'n_estimators': 400,
                    'num_leaves': 31,
                    'learning_rate': 0.02,
                    'feature_fraction': 0.8,
                    'bagging_fraction': 0.8,
                    'bagging_freq': 5,
                    'reg_alpha': 0.1,
                    'reg_lambda': 0.1,
                    'min_child_samples': 20,
                    'random_state': 42,
                    'n_jobs': -1,
                    'verbose': -1,
                    'metric': 'rmse'
                }
            }
        }
    
    def train_model_with_cv(self, X_train, y_train, model_config, n_folds=5):
        """
        Train a model using cross-validation and return predictions and performance metrics.
        
        Parameters:
        -----------
        X_train : pandas.DataFrame
            Training features
        y_train : pandas.Series
            Training target
        model_config : dict
            Model configuration including class and parameters
        n_folds : int
            Number of cross-validation folds
        
        Returns:
        --------
        tuple
            (trained_model, cv_scores, feature_importance)
        """
        model_name = model_config['model_class'].__name__
        print(f"\nTraining {model_name} with {n_folds}-fold CV...")
        
        # Initialize model
        model_class = model_config['model_class']
        params = model_config['params']
        
        # Prepare for cross-validation
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        cv_scores = []
        feature_importance = np.zeros(len(X_train.columns))
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        
        # Cross-validation
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled)):
            X_fold_train = X_scaled[train_idx]
            y_fold_train = y_train.iloc[train_idx]
            X_fold_val = X_scaled[val_idx]
            y_fold_val = y_train.iloc[val_idx]
            
            # Train model
            model = model_class(**params)
            
            # Fit with early stopping for tree-based models
            if model_name in ['XGBRegressor', 'LGBMRegressor']:
                model.fit(
                    X_fold_train, 
                    y_fold_train,
                    eval_set=[(X_fold_val, y_fold_val)],
                    eval_metric='rmse',
                    early_stopping_rounds=50,
                    verbose=False
                )
            else:
                model.fit(X_fold_train, y_fold_train)
            
            # Validate
            val_predictions = model.predict(X_fold_val)
            
            # Calculate correlation
            try:
                correlation = pearsonr(y_fold_val, val_predictions)[0]
                if np.isnan(correlation):
                    correlation = 0.0
            except:
                correlation = 0.0
                
            cv_scores.append(correlation)
            
            # Accumulate feature importance
            if hasattr(model, 'feature_importances_'):
                feature_importance += model.feature_importances_ / n_folds
            
            print(f"  Fold {fold + 1}: Correlation = {correlation:.6f}")
        
        # Train final model on full data
        print("  Training final model on full dataset...")
        final_model = model_class(**params)
        
        if model_name in ['XGBRegressor', 'LGBMRegressor']:
            # Use a portion for validation
            val_size = int(0.1 * len(X_scaled))
            X_train_final = X_scaled[:-val_size]
            y_train_final = y_train.iloc[:-val_size]
            X_val_final = X_scaled[-val_size:]
            y_val_final = y_train.iloc[-val_size:]
            
            final_model.fit(
                X_train_final, 
                y_train_final,
                eval_set=[(X_val_final, y_val_final)],
                eval_metric='rmse',
                early_stopping_rounds=50,
                verbose=False
            )
        else:
            final_model.fit(X_scaled, y_train)
        
        # Store scaler with model for later use
        final_model.scaler = scaler
        final_model.feature_names = list(X_train.columns)
        
        mean_score = np.mean(cv_scores)
        std_score = np.std(cv_scores)
        print(f"  Mean CV Correlation: {mean_score:.6f} (±{std_score:.6f})")
        
        return final_model, cv_scores, feature_importance
    
    def generate_predictions(self, model, X_test):
        """Generate predictions for test data using a trained model."""
        # Apply the same scaling used during training
        X_test_scaled = model.scaler.transform(X_test)
        predictions = model.predict(X_test_scaled)
        
        return predictions
    
    def create_ensemble_predictions(self, predictions_dict, weights=None):
        """
        Create ensemble predictions from multiple models.
        
        Parameters:
        -----------
        predictions_dict : dict
            Dictionary of model predictions
        weights : dict or None
            Optional weights for each model (defaults to equal weights)
        
        Returns:
        --------
        numpy.ndarray
            Ensemble predictions
        """
        if weights is None:
            weights = {name: 1.0 / len(predictions_dict) for name in predictions_dict}
        
        # Ensure weights sum to 1
        total_weight = sum(weights.values())
        weights = {k: v/total_weight for k, v in weights.items()}
        
        ensemble_pred = np.zeros_like(next(iter(predictions_dict.values())))
        
        for name, pred in predictions_dict.items():
            ensemble_pred += weights.get(name, 0) * pred
        
        return ensemble_pred
    
    def save_submission(self, predictions, test_ids, filename):
        """Save predictions in competition submission format."""
        submission = pd.DataFrame({
            'id': test_ids,
            'prediction': predictions
        })
        
        submission.to_csv(filename, index=False)
        print(f"Saved submission to {filename}")
        
        # Display sample predictions
        print(f"  Sample predictions: {predictions[:5]}")
        print(f"  Prediction statistics: mean={np.mean(predictions):.6f}, std={np.std(predictions):.6f}")
        
        return submission
    
    def run_complete_pipeline(self, use_sample=False, sample_size=100000):
        """
        Execute the complete training and submission generation pipeline.
        
        Parameters:
        -----------
        use_sample : bool
            Whether to use a sample of the data for faster iteration
        sample_size : int
            Number of rows to sample if use_sample is True
        """
        print("="*80)
        print("CRYPTO MARKET PREDICTION MODEL TRAINING")
        print("="*80)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Step 1: Load SULOV results and prepare feature sets
        print("\nStep 1: Preparing feature sets...")
        
        # First, load a small sample to get all feature names
        print("Reading feature names from training data...")
        train_sample = pd.read_parquet(self.train_path)
        all_features = [col for col in train_sample.columns if col not in ['timestamp', 'label']]
        print(f"Total features available: {len(all_features)}")
        
        # Clean up sample
        del train_sample
        gc.collect()
        
        # Get SULOV features
        top_30_sulov, feature_correlations = self.load_sulov_results()
        
        # If no SULOV features found, use a default set
        if not top_30_sulov:
            print("Warning: No SULOV features found, using first 30 anonymized features")
            anonymized = [f for f in all_features if f.startswith('X')]
            top_30_sulov = anonymized[:30]
        
        sulov_features = self.baseline_features + top_30_sulov
        
        # Get random features
        random_30 = self.select_random_features(all_features)
        random_features = self.baseline_features + random_30
        
        # Store feature sets
        self.feature_sets = {
            'sulov': sulov_features,
            'random': random_features
        }
        
        # Print feature set summaries
        print(f"\nSULOV feature set: {len(sulov_features)} features")
        print(f"  Baseline: {self.baseline_features}")
        print(f"  Top 5 SULOV: {top_30_sulov[:5]}")
        
        print(f"\nRandom feature set: {len(random_features)} features")
        print(f"  Baseline: {self.baseline_features}")
        print(f"  First 5 random: {random_30[:5]}")
        
        # Step 2: Load data for each feature set and train models
        print("\nStep 2: Loading data and training models...")
        
        model_configs = self.create_model_configurations()
        
        # Determine sample size for training
        train_sample_size = sample_size if use_sample else None
        
        for feature_set_name, features in self.feature_sets.items():
            print(f"\n{'='*60}")
            print(f"Training models with {feature_set_name.upper()} features")
            print(f"{'='*60}")
            
            # Load data
            X_train, y_train, X_test, test_ids = self.load_and_prepare_data(
                features, 
                sample_size=train_sample_size
            )
            
            # Train different model configurations
            set_predictions = {}
            
            for model_name, model_config in model_configs.items():
                model_key = f"{feature_set_name}_{model_name}"
                
                try:
                    # Train model
                    model, cv_scores, feature_importance = self.train_model_with_cv(
                        X_train, y_train, model_config
                    )
                    
                    # Generate predictions
                    predictions = self.generate_predictions(model, X_test)
                    
                    # Store results
                    self.models[model_key] = model
                    self.predictions[model_key] = predictions
                    self.performance_metrics[model_key] = {
                        'cv_scores': cv_scores,
                        'mean_cv': np.mean(cv_scores),
                        'std_cv': np.std(cv_scores),
                        'feature_importance': feature_importance
                    }
                    
                    set_predictions[model_name] = predictions
                    
                except Exception as e:
                    print(f"  Error training {model_key}: {str(e)}")
                    continue
            
            # Create ensemble for this feature set
            if set_predictions:
                ensemble_key = f"{feature_set_name}_ensemble"
                ensemble_predictions = self.create_ensemble_predictions(set_predictions)
                self.predictions[ensemble_key] = ensemble_predictions
                print(f"\nCreated ensemble for {feature_set_name} with {len(set_predictions)} models")
        
        # Step 3: Generate submissions
        print("\n" + "="*80)
        print("Step 3: Generating submission files...")
        print("="*80)
        
        # Ensure we have test_ids
        if 'test_ids' not in locals():
            _, _, _, test_ids = self.load_and_prepare_data(self.baseline_features)
        
        # Submission 1: Best SULOV model (highest CV score)
        sulov_models = {k: v for k, v in self.performance_metrics.items() if k.startswith('sulov_')}
        
        if sulov_models:
            best_sulov_model = max(sulov_models.keys(), key=lambda x: sulov_models[x]['mean_cv'])
            print(f"\nBest SULOV model: {best_sulov_model}")
            print(f"  CV Score: {sulov_models[best_sulov_model]['mean_cv']:.6f}")
            
            submission_1 = self.save_submission(
                self.predictions[best_sulov_model],
                test_ids,
                f"submission_1_best_sulov_{timestamp}.csv"
            )
        
        # Submission 2: SULOV ensemble
        if 'sulov_ensemble' in self.predictions:
            print("\nGenerating SULOV ensemble submission...")
            submission_2 = self.save_submission(
                self.predictions['sulov_ensemble'],
                test_ids,
                f"submission_2_sulov_ensemble_{timestamp}.csv"
            )
        
        # Submission 3: Mixed ensemble (SULOV + Random)
        if 'sulov_ensemble' in self.predictions and 'random_ensemble' in self.predictions:
            print("\nGenerating mixed ensemble submission...")
            mixed_ensemble = self.create_ensemble_predictions({
                'sulov': self.predictions['sulov_ensemble'],
                'random': self.predictions['random_ensemble']
            }, weights={'sulov': 0.7, 'random': 0.3})
            
            submission_3 = self.save_submission(
                mixed_ensemble,
                test_ids,
                f"submission_3_mixed_ensemble_{timestamp}.csv"
            )
        
        # Step 4: Generate performance report
        self.generate_performance_report(timestamp)
        
        print("\n" + "="*80)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("="*80)
        
        return True
    
    def generate_performance_report(self, timestamp):
        """Generate a comprehensive performance report."""
        report_path = f"model_performance_report_{timestamp}.txt"
        
        with open(report_path, 'w') as f:
            f.write("CRYPTO MARKET PREDICTION MODEL PERFORMANCE REPORT\n")
            f.write("="*80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Feature set comparison
            f.write("FEATURE SET COMPARISON\n")
            f.write("-"*40 + "\n")
            
            # Calculate average performance by feature set
            for feature_set in ['sulov', 'random']:
                set_metrics = [(k, v) for k, v in self.performance_metrics.items() 
                             if k.startswith(f"{feature_set}_")]
                
                if set_metrics:
                    avg_cv = np.mean([m[1]['mean_cv'] for m in set_metrics])
                    
                    f.write(f"\n{feature_set.upper()} Features:\n")
                    f.write(f"  Number of features: {len(self.feature_sets.get(feature_set, []))}\n")
                    f.write(f"  Average CV correlation: {avg_cv:.6f}\n")
                    
                    if feature_set in self.feature_sets:
                        feature_preview = self.feature_sets[feature_set][:10]
                        f.write(f"  First 10 features: {', '.join(feature_preview)}\n")
            
            # Individual model performance
            f.write("\n\nINDIVIDUAL MODEL PERFORMANCE\n")
            f.write("-"*40 + "\n")
            
            if self.performance_metrics:
                sorted_models = sorted(self.performance_metrics.items(), 
                                     key=lambda x: x[1]['mean_cv'], reverse=True)
                
                for model_name, metrics in sorted_models:
                    f.write(f"\n{model_name}:\n")
                    f.write(f"  Mean CV Correlation: {metrics['mean_cv']:.6f}\n")
                    f.write(f"  Std CV Correlation: {metrics['std_cv']:.6f}\n")
                    f.write(f"  CV Scores: {[f'{s:.6f}' for s in metrics['cv_scores']]}\n")
            
            # Feature importance for top models
            f.write("\n\nTOP FEATURE IMPORTANCE\n")
            f.write("-"*40 + "\n")
            
            # Get best model from each feature set
            for feature_set in ['sulov', 'random']:
                set_models = {k: v for k, v in self.performance_metrics.items() 
                            if k.startswith(f"{feature_set}_")}
                
                if set_models:
                    best_model_key = max(set_models.keys(), 
                                       key=lambda x: set_models[x]['mean_cv'])
                    
                    if best_model_key in self.models:
                        model = self.models[best_model_key]
                        importance = self.performance_metrics[best_model_key].get('feature_importance', [])
                        
                        if len(importance) > 0 and hasattr(model, 'feature_names'):
                            f.write(f"\n{best_model_key} - Top 10 features:\n")
                            
                            # Get feature importance with names
                            feature_imp = list(zip(model.feature_names, importance))
                            feature_imp.sort(key=lambda x: x[1], reverse=True)
                            
                            for i, (feat, imp) in enumerate(feature_imp[:10]):
                                f.write(f"  {i+1}. {feat}: {imp:.4f}\n")
            
            # Submission descriptions
            f.write("\n\nSUBMISSION DESCRIPTIONS\n")
            f.write("-"*40 + "\n")
            f.write("\nSubmission 1: Best performing SULOV-based model\n")
            f.write("  - Uses features selected through SULOV clustering algorithm\n")
            f.write("  - Single model with highest cross-validation performance\n")
            
            f.write("\nSubmission 2: Ensemble of all SULOV-based models\n")
            f.write("  - Combines predictions from XGBoost and LightGBM variants\n")
            f.write("  - Equal weighting of all models\n")
            
            f.write("\nSubmission 3: Mixed ensemble (70% SULOV, 30% Random)\n")
            f.write("  - Weighted combination of SULOV and random feature models\n")
            f.write("  - Provides robustness against feature selection bias\n")
        
        print(f"\nPerformance report saved to {report_path}")


def main():
    """Execute the complete model training and submission generation pipeline."""
    # Initialize predictor
    predictor = CryptoMarketPredictor(
        train_path="/kaggle/input/drw-crypto-market-prediction/train.parquet",
        test_path="/kaggle/input/drw-crypto-market-prediction/test.parquet",
        sulov_results_path="sulov_selection_results.json"
    )
    
    # Run complete pipeline
    # Set use_sample=True and adjust sample_size for faster testing
    success = predictor.run_complete_pipeline(use_sample=False)
    
    if success:
        # Display final summary
        print("\nFinal Summary:")
        print("-"*40)
        print("Three submission files have been generated:")
        print("1. Best SULOV model - Single model with highest CV performance")
        print("2. SULOV ensemble - Combination of all SULOV-based models")
        print("3. Mixed ensemble - Weighted combination of SULOV and random features")
        print("\nReview the performance report for detailed metrics and comparisons.")
        
        # Display model performance summary
        if predictor.performance_metrics:
            print("\nModel Performance Summary:")
            print("-"*40)
            
            # Calculate average performance by feature set
            for feature_set in ['sulov', 'random']:
                models = [(k, v['mean_cv']) for k, v in predictor.performance_metrics.items() 
                         if k.startswith(f"{feature_set}_")]
                
                if models:
                    avg_score = np.mean([score for _, score in models])
                    best_model = max(models, key=lambda x: x[1])
                    
                    print(f"\n{feature_set.upper()} Features:")
                    print(f"  Average CV Score: {avg_score:.6f}")
                    print(f"  Best Model: {best_model[0]} (Score: {best_model[1]:.6f})")


if __name__ == "__main__":
    main()

