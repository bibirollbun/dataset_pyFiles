import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import catboost as cb
import warnings
warnings.filterwarnings('ignore')




class ArielEnsembleModel:
    def __init__(self, fast_mode=False):
        """
        Initialize the ensemble model with XGBoost and CatBoost
        fast_mode: If True, use faster parameters for Kaggle submission
        """
        self.fast_mode = fast_mode  # <--- Added this line

        # XGBoost parameters - TUNABLE PARAMETERS MARKED WITH COMMENTS
        if self.fast_mode: # Changed to self.fast_mode
            # Faster parameters for Kaggle submission
            self.xgb_params = {
                'objective': 'reg:squarederror',
                'n_estimators': 300,   # Reduced for faster training
                'max_depth': 6,        # Reduced for faster training
                'learning_rate': 0.1,  # Increased for faster convergence
                'subsample': 0.8,      # TUNABLE: Try 0.6-0.9
                'colsample_bytree': 0.8, # TUNABLE: Try 0.6-0.9
                'reg_alpha': 0.1,      # TUNABLE: Try 0.0-1.0 (L1 regularization)
                'reg_lambda': 0.1,     # TUNABLE: Try 0.0-1.0 (L2 regularization)
                'min_child_weight': 1, # TUNABLE: Try 1-10
                'gamma': 0,            # TUNABLE: Try 0-0.5
                'random_state': 42,
                'n_jobs': -1,
                'tree_method': 'hist', # Faster tree method
                'verbose': 0
            }
        else:
            # Full parameters for better accuracy
            self.xgb_params = {
                'objective': 'reg:squarederror',
                'n_estimators': 1000,  # TUNABLE: Try 500-3000
                'max_depth': 8,        # TUNABLE: Try 4-12
                'learning_rate': 0.05, # TUNABLE: Try 0.01-0.3
                'subsample': 0.8,      # TUNABLE: Try 0.6-0.9
                'colsample_bytree': 0.8, # TUNABLE: Try 0.6-0.9
                'reg_alpha': 0.1,      # TUNABLE: Try 0.0-1.0 (L1 regularization)
                'reg_lambda': 0.1,     # TUNABLE: Try 0.0-1.0 (L2 regularization)
                'min_child_weight': 1, # TUNABLE: Try 1-10
                'gamma': 0,            # TUNABLE: Try 0-0.5
                'random_state': 42,
                'n_jobs': -1,
                'tree_method': 'hist',
                'early_stopping_rounds': 50,
                'verbose': 0
            }

        # CatBoost parameters - TUNABLE PARAMETERS MARKED WITH COMMENTS
        if self.fast_mode: # Changed to self.fast_mode
            # Faster parameters for Kaggle submission
            self.cb_params = {
                'iterations': 300,     # Reduced for faster training
                'depth': 6,            # Reduced for faster training
                'learning_rate': 0.1,  # Increased for faster convergence
                'l2_leaf_reg': 3,      # TUNABLE: Try 1-10
                'random_strength': 1,  # TUNABLE: Try 0-10
                'bootstrap_type': 'Bernoulli', # TUNABLE: Try 'Bernoulli', 'MVS', 'Poisson'
                'subsample': 0.8,      # TUNABLE: Try 0.6-0.9
                'random_state': 42,
                'verbose': 0,
                'allow_writing_files': False,  # Important for Kaggle
                'thread_count': -1
            }
        else:
            # Full parameters for better accuracy
            self.cb_params = {
                'iterations': 1000,     # TUNABLE: Try 500-3000
                'depth': 8,            # TUNABLE: Try 4-12
                'learning_rate': 0.05, # TUNABLE: Try 0.01-0.3
                'l2_leaf_reg': 3,      # TUNABLE: Try 1-10
                'random_strength': 1,   # TUNABLE: Try 0-10
                'bootstrap_type': 'Bernoulli', # TUNABLE: Try 'Bernoulli', 'MVS', 'Poisson'
                'subsample': 0.8,       # TUNABLE: Try 0.6-0.9
                'random_state': 42,
                'verbose': 0,
                'early_stopping_rounds': 50,
                'allow_writing_files': False,  # Important for Kaggle
                'thread_count': -1
            }

        # Alternative CatBoost configurations for different bootstrap types
        # Uncomment one of these if you want to try different bootstrap methods:

        # Option 1: Bayesian bootstrap (no subsample, has bagging_temperature)
        # self.cb_params = {
        #     'iterations': 1000,
        #     'depth': 8,
        #     'learning_rate': 0.05,
        #     'l2_leaf_reg': 3,
        #     'bagging_temperature': 1,  # Only for Bayesian
        #     'random_strength': 1,
        #     'bootstrap_type': 'Bayesian',
        #     'random_state': 42,
        #     'verbose': 0,
        #     'early_stopping_rounds': 50
        # }

        # Option 2: MVS bootstrap (has subsample, no bagging_temperature)
        # self.cb_params = {
        #     'iterations': 1000,
        #     'depth': 8,
        #     'learning_rate': 0.05,
        #     'l2_leaf_reg': 3,
        #     'random_strength': 1,
        #     'bootstrap_type': 'MVS',
        #     'subsample': 0.8,
        #     'random_state': 42,
        #     'verbose': 0,
        #     'early_stopping_rounds': 50
        # }
        self.xgb_weight = 0.5  # TUNABLE: Try 0.3-0.7
        self.cb_weight = 0.5   # TUNABLE: Try 0.3-0.7

        self.models = {}
        self.scalers = {}
        self.target_cols = []

    def load_data(self, data_path='/kaggle/input/ariel-data-challenge-2025/'):
        """
        Load and prepare the dataset
        """
        print("Loading data...")

        # Load main datasets
        self.train_df = pd.read_csv(f'{data_path}train.csv')
        self.test_df = pd.read_csv(f'{data_path}test_star_info.csv')

        # Load auxiliary data
        self.train_star_info = pd.read_csv(f'{data_path}train_star_info.csv')
        self.wavelengths = pd.read_csv(f'{data_path}wavelengths.csv')
        self.adc_info = pd.read_csv(f'{data_path}adc_info.csv')

        print(f"Train shape: {self.train_df.shape}")
        print(f"Test shape: {self.test_df.shape}")
        print(f"Train star info shape: {self.train_star_info.shape}")
        print(f"Wavelengths shape: {self.wavelengths.shape}")

        # Get target columns (all columns except planet_id in train.csv)
        self.target_cols = [col for col in self.train_df.columns if col != 'planet_id']
        print(f"Number of target columns: {len(self.target_cols)}")

    def feature_engineering(self):
        """
        Create additional features from star information and wavelengths
        """
        print("Performing feature engineering...")

        # Merge star information with train and test data
        self.train_features = self.train_star_info.copy()
        self.test_features = self.test_df.copy()

        # Create additional features from star parameters
        for df in [self.train_features, self.test_features]:
            # Stellar ratios and combinations - TUNABLE: Add more domain-specific features
            df['Rs_Ms_ratio'] = df['Rs'] / (df['Ms'] + 1e-8)
            df['Ts_Mp_ratio'] = df['Ts'] / (df['Mp'] + 1e-8)
            df['e_P_product'] = df['e'] * df['P']
            df['sma_P_ratio'] = df['sma'] / (df['P'] + 1e-8)
            df['stellar_luminosity'] = df['Rs']**2 * (df['Ts']/5778)**4  # Relative to Sun
            df['planet_insolation'] = df['stellar_luminosity'] / (df['sma']**2 + 1e-8)
            df['equilibrium_temp'] = df['Ts'] * np.sqrt(df['Rs']/(2*df['sma'] + 1e-8))

            # Polynomial features for important parameters - TUNABLE: Adjust degree
            df['Ts_squared'] = df['Ts']**2
            df['Rs_squared'] = df['Rs']**2
            df['Mp_squared'] = df['Mp']**2
            df['log_P'] = np.log(df['P'] + 1e-8)
            df['log_sma'] = np.log(df['sma'] + 1e-8)

        # Statistical features from wavelength data - TUNABLE: Add more statistical features
        wavelength_stats = self.wavelengths.describe().T
        for stat in ['mean', 'std', 'min', 'max']:
            self.train_features[f'wavelength_{stat}'] = wavelength_stats[stat].iloc[0]
            self.test_features[f'wavelength_{stat}'] = wavelength_stats[stat].iloc[0]

        # Remove planet_id for modeling
        feature_cols = [col for col in self.train_features.columns if col != 'planet_id']
        self.X_train = self.train_features[feature_cols]
        self.X_test = self.test_features[feature_cols]

        print(f"Feature engineering complete. Total features: {len(feature_cols)}")

    def train_models(self):
        """
        Train XGBoost and CatBoost models for each target
        """
        print("Training models...")

        # Prepare targets
        y_train = self.train_df[self.target_cols]

        # Use smaller validation split for fast mode
        test_size = 0.1 if self.fast_mode else 0.2 # Changed to self.fast_mode

        # Split for validation
        X_train, X_val, y_train_split, y_val_split = train_test_split(
            self.X_train, y_train, test_size=test_size, random_state=42
        )

        # Scale features - TUNABLE: Try different scalers or no scaling
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(self.X_test)

        # Train models for each target
        self.xgb_models = {}
        self.cb_models = {}

        # Use early stopping for non-fast mode
        use_early_stopping = not self.fast_mode # Changed to self.fast_mode

        for i, target_col in enumerate(self.target_cols):
            if i % 50 == 0:
                print(f"Training models for target {i+1}/{len(self.target_cols)}: {target_col}")

            # Train XGBoost
            xgb_model = xgb.XGBRegressor(**self.xgb_params)
            if use_early_stopping:
                xgb_model.fit(
                    X_train_scaled, y_train_split[target_col],
                    eval_set=[(X_val_scaled, y_val_split[target_col])],
                    verbose=False
                )
            else:
                xgb_model.fit(X_train_scaled, y_train_split[target_col])
            self.xgb_models[target_col] = xgb_model

            # Train CatBoost
            cb_model = cb.CatBoostRegressor(**self.cb_params)
            if use_early_stopping:
                cb_model.fit(
                    X_train_scaled, y_train_split[target_col],
                    eval_set=(X_val_scaled, y_val_split[target_col]),
                    verbose=False
                )
            else:
                cb_model.fit(X_train_scaled, y_train_split[target_col])
            self.cb_models[target_col] = cb_model

        # Store scaled test features
        self.X_test_scaled = X_test_scaled

        print("Model training complete!")

    def predict(self):
        """
        Make predictions using ensemble of XGBoost and CatBoost
        """
        print("Making predictions...")

        predictions = {}

        for target_col in self.target_cols:
            # Get predictions from both models
            xgb_pred = self.xgb_models[target_col].predict(self.X_test_scaled)
            cb_pred = self.cb_models[target_col].predict(self.X_test_scaled)

            # Ensemble predictions - TUNABLE: Try different ensemble methods
            ensemble_pred = (self.xgb_weight * xgb_pred + self.cb_weight * cb_pred)
            predictions[target_col] = ensemble_pred

        return predictions

    def evaluate_models(self):
        """
        Evaluate model performance using cross-validation
        """
        print("Evaluating models...")

        # Sample a few targets for evaluation (to save time)
        sample_targets = self.target_cols[:10]  # Evaluate first 10 targets

        cv_scores = {}
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)

        # Create parameter sets for cross-validation, explicitly removing early_stopping_rounds
        # since cross_val_score doesn't provide an eval_set for it.
        xgb_cv_params = self.xgb_params.copy()
        if 'early_stopping_rounds' in xgb_cv_params:
            del xgb_cv_params['early_stopping_rounds']
        if 'verbose' in xgb_cv_params and xgb_cv_params['verbose'] == 0: # Ensure verbose is not 0 for potential warnings if needed, but for CV typically kept silent
             del xgb_cv_params['verbose'] # Remove verbose for clean CV output if 0

        cb_cv_params = self.cb_params.copy()
        if 'early_stopping_rounds' in cb_cv_params:
            del cb_cv_params['early_stopping_rounds']
        if 'verbose' in cb_cv_params and cb_cv_params['verbose'] == 0: # Same for CatBoost
            del cb_cv_params['verbose'] # Remove verbose if 0 for clean CV output


        for target_col in sample_targets:
            y_target = self.train_df[target_col]

            # XGBoost CV
            xgb_model = xgb.XGBRegressor(**xgb_cv_params) # Use the modified params
            xgb_scores = cross_val_score(
                xgb_model, self.X_train, y_target,
                cv=kfold, scoring='neg_mean_squared_error'
            )

            # CatBoost CV
            cb_model = cb.CatBoostRegressor(**cb_cv_params) # Use the modified params
            cb_scores = cross_val_score(
                cb_model, self.X_train, y_target,
                cv=kfold, scoring='neg_mean_squared_error'
            )

            cv_scores[target_col] = {
                'xgb_rmse': np.sqrt(-xgb_scores.mean()),
                'cb_rmse': np.sqrt(-cb_scores.mean()),
                'xgb_std': np.sqrt(xgb_scores.std()),
                'cb_std': np.sqrt(cb_scores.std())
            }

        # Print evaluation results
        print("\nCross-validation Results (Sample):")
        print("=" * 60)
        for target, scores in cv_scores.items():
            print(f"{target[:20]:20} | XGB: {scores['xgb_rmse']:.4f}±{scores['xgb_std']:.4f} | "
                  f"CB: {scores['cb_rmse']:.4f}±{scores['cb_std']:.4f}")

        return cv_scores

    def create_submission(self, predictions, filename='submission.csv'):
        """
        Create submission file in the required format
        """
        print(f"Creating submission file: {filename}")

        # Create submission dataframe
        submission_df = pd.DataFrame()
        submission_df['planet_id'] = self.test_features['planet_id']

        # Add predictions for all targets
        for target_col in self.target_cols:
            submission_df[target_col] = predictions[target_col]

        # Save submission
        submission_df.to_csv(filename, index=False)
        print(f"Submission file saved with shape: {submission_df.shape}")

        return submission_df


def main():
    """
    Main execution function
    """
    import os
    
    # Use fast mode for Kaggle submission to avoid timeout
    fast_mode = os.environ.get('KAGGLE_KERNEL_RUN_TYPE') == 'Interactive'
    
    # Initialize model
    model = ArielEnsembleModel(fast_mode=fast_mode)
    
    if fast_mode:
        print("Running in FAST MODE for Kaggle submission")
    else:
        print("Running in FULL MODE for better accuracy")
    
    # Load data
    model.load_data()
    
    # Feature engineering
    model.feature_engineering()
    
    # Evaluate models (optional - skip in fast mode)
    if not fast_mode:
        model.evaluate_models()
    
    # Train models
    model.train_models()
    
    # Make predictions
    predictions = model.predict()
    
    # Create submission
    submission = model.create_submission(predictions)
    
    print("Pipeline completed successfully!")
    
    # Display feature importance for first target (XGBoost)
    if len(model.target_cols) > 0:
        first_target = model.target_cols[0]
        importance = model.xgb_models[first_target].feature_importances_
        feature_names = [col for col in model.X_train.columns]
        
        print(f"\nTop 10 Feature Importances for {first_target}:")
        print("=" * 50)
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        for i, row in importance_df.head(10).iterrows():
            print(f"{row['feature']:30} | {row['importance']:.4f}")
def run_kaggle_submission():
    """
    Optimized function for Kaggle submission
    """
    # Initialize model in fast mode
    model = ArielEnsembleModel(fast_mode=True)
    
    # Load data (adjust path if needed)
    model.load_data('/kaggle/input/ariel-data-challenge-2025/')
    
    # Feature engineering
    model.feature_engineering()
    
    # Train models
    model.train_models()
    
    # Make predictions
    predictions = model.predict()
    
    # Create submission
    submission = model.create_submission(predictions, '/kaggle/working/submission.csv')
    
    print("Kaggle submission ready!")
    return submission

if __name__ == "__main__":
    main()


