from IPython.display import Image, display

img_path = "/kaggle/input/kaggle-binary-classification-logo/logo.png"

display(Image(filename=img_path))


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


# ================================================================================
# KAGGLE PLAYGROUND SERIES S5E8 
# ================================================================================
# Improved pipeline with better structure, feature engineering, and ensemble methods
# Author: Imaad Mahmood
# Date: August 11, 2025
# ================================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ================================================================================
# MACHINE LEARNING LIBRARIES
# ================================================================================
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.base import BaseEstimator, TransformerMixin

# Model libraries
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression

# Utility libraries
from datetime import datetime
import gc

# ================================================================================
# CONFIGURATION AND GLOBAL SETTINGS
# ================================================================================
CONFIG = {
    'random_state': 42,
    'n_folds': 5,
    'target_col': 'y',
    'id_col': 'id',
    'data_path': '/kaggle/input/playground-series-s5e8/',
    'verbose': True
}

# Set random seeds for reproducibility
np.random.seed(CONFIG['random_state'])

# ================================================================================
# DATA LOADING AND INITIAL EXPLORATION
# ================================================================================
def load_data():
    """Load train, test, and submission data"""
    print("=" * 60)
    print("LOADING DATA...")
    print("=" * 60)
    
    train = pd.read_csv(f"{CONFIG['data_path']}train.csv")
    test = pd.read_csv(f"{CONFIG['data_path']}test.csv")
    sample_submission = pd.read_csv(f"{CONFIG['data_path']}sample_submission.csv")
    
    print(f"Train Shape: {train.shape}")
    print(f"Test Shape: {test.shape}")
    print(f"Submission Shape: {sample_submission.shape}")
    
    # Target distribution analysis
    target_dist = train[CONFIG['target_col']].value_counts()
    print(f"\nTarget Distribution:")
    print(f"Class 0: {target_dist[0]:,} ({target_dist[0]/len(train)*100:.2f}%)")
    print(f"Class 1: {target_dist[1]:,} ({target_dist[1]/len(train)*100:.2f}%)")
    
    return train, test, sample_submission

# ================================================================================
# ADVANCED FEATURE ENGINEERING CLASS
# ================================================================================
class AdvancedFeatureEngineer(BaseEstimator, TransformerMixin):
    """Advanced feature engineering pipeline"""
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def fit(self, X, y=None):
        """Fit the feature engineering pipeline"""
        X = X.copy()
        
        # ====== REMOVE TARGET COLUMN IF EXISTS ======
        if CONFIG['target_col'] in X.columns:
            X = X.drop(CONFIG['target_col'], axis=1)
        
        # ====== CATEGORICAL ENCODING ======
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            le.fit(X[col].astype(str))
            self.label_encoders[col] = le
            
        # ====== NUMERICAL SCALING PREPARATION ======
        X_processed = self._create_features(X)
        numerical_cols = X_processed.select_dtypes(exclude=['object']).columns
        self.scaler.fit(X_processed[numerical_cols])
        
        self.feature_names = X_processed.columns.tolist()
        
        return self
    
    def transform(self, X):
        """Transform the data with feature engineering"""
        X = X.copy()
        
        # ====== REMOVE TARGET COLUMN IF EXISTS ======
        if CONFIG['target_col'] in X.columns:
            X = X.drop(CONFIG['target_col'], axis=1)
            
        X_processed = self._create_features(X)
        
        # ====== APPLY SCALING ======
        numerical_cols = X_processed.select_dtypes(exclude=['object']).columns
        X_processed[numerical_cols] = self.scaler.transform(X_processed[numerical_cols])
        
        return X_processed
    
    def _create_features(self, X):
        """Core feature engineering logic"""
        X = X.copy()
        
        # ====== HANDLE ID COLUMN ======
        if CONFIG['id_col'] in X.columns:
            X = X.drop(CONFIG['id_col'], axis=1)
            
        # ====== ENCODE ORIGINAL CATEGORICAL VARIABLES FIRST ======
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col in self.label_encoders:
                X[col] = self.label_encoders[col].transform(X[col].astype(str))
            else:
                # Handle unseen categories during fitting
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                
        # ====== CONTACT HISTORY FEATURES ======
        # More sophisticated handling of previous contact information
        X['was_contacted_before'] = (X['pdays'] != -1).astype(int)
        X['pdays_log'] = np.where(X['pdays'] == -1, 0, np.log1p(X['pdays']))
        X['previous_log'] = np.log1p(X['previous'])
        
        # Contact efficiency ratio (using encoded poutcome)
        # Assuming 'success' maps to highest encoded value for poutcome
        max_poutcome = X['poutcome'].max()
        X['success_rate'] = np.where(X['previous'] > 0, 
                                   (X['poutcome'] == max_poutcome).astype(int) / (X['previous'] + 1), 
                                   0)
        
        # ====== FINANCIAL FEATURES ======
        # Advanced balance handling
        X['balance_is_negative'] = (X['balance'] < 0).astype(int)
        X['balance_abs'] = X['balance'].abs()
        X['balance_log'] = np.log1p(X['balance_abs'])
        
        # Financial status as numerical categories
        X['financial_status'] = pd.cut(X['balance'], 
                                     bins=[-np.inf, -1000, 0, 1000, 5000, np.inf],
                                     labels=[0, 1, 2, 3, 4]).astype(int)
        
        # ====== DEMOGRAPHIC FEATURES ======
        # Age groups as numerical categories
        X['age_group'] = pd.cut(X['age'], 
                              bins=[0, 25, 35, 45, 55, 65, 100],
                              labels=[0, 1, 2, 3, 4, 5]).astype(int)
        
        # Education level ordering (assuming hierarchy)
        education_order = {0: 0, 1: 1, 2: 2, 3: 3}  # Using encoded values
        X['education_level'] = X['education'].map(education_order).fillna(0).astype(int)
        
        # ====== CAMPAIGN FEATURES ======
        # Campaign intensity and effectiveness
        X['campaign_log'] = np.log1p(X['campaign'])
        X['campaign_intensity'] = pd.cut(X['campaign'], 
                                       bins=[0, 1, 3, 5, 10, 100],
                                       labels=[0, 1, 2, 3, 4]).astype(int)
        
        # ====== TEMPORAL FEATURES ======
        # Month seasonality as numerical (using encoded month values)
        # Create season mapping based on month ranges
        def month_to_season(month_encoded):
            # Assuming months are encoded 0-11 or similar pattern
            if month_encoded in [11, 0, 1]:  # Dec, Jan, Feb
                return 0  # winter
            elif month_encoded in [2, 3, 4]:  # Mar, Apr, May
                return 1  # spring
            elif month_encoded in [5, 6, 7]:  # Jun, Jul, Aug
                return 2  # summer
            else:  # Sep, Oct, Nov
                return 3  # autumn
                
        X['season'] = X['month'].apply(month_to_season)
        
        # Call duration features
        X['duration_log'] = np.log1p(X['duration'])
        X['duration_category'] = pd.cut(X['duration'], 
                                      bins=[0, 60, 180, 300, 600, np.inf],
                                      labels=[0, 1, 2, 3, 4]).astype(int)
        
        # ====== INTERACTION FEATURES ======
        # Job-education interaction (using encoded values)
        # Find encoded value for 'student' job and 'tertiary' education
        X['job_education_high'] = ((X['job'] == X['job'].mode()[0]) & 
                                  (X['education'] == X['education'].max())).astype(int)
        
        # Housing loan interaction (using encoded values)
        X['housing_loan_both'] = ((X['housing'] == 1) & (X['loan'] == 1)).astype(int)
        
        # Age-job interaction for high-level jobs
        X['senior_management'] = ((X['age'] >= 50) & 
                                 (X['job'] == X['job'].mode()[0])).astype(int)
        
        # Additional financial interactions
        X['high_balance_long_call'] = ((X['balance_abs'] > X['balance_abs'].quantile(0.8)) & 
                                     (X['duration'] > X['duration'].quantile(0.8))).astype(int)
        
        return X

# ================================================================================
# MODEL CONFIGURATIONS WITH OPTIMIZED HYPERPARAMETERS
# ================================================================================
MODEL_CONFIGS = {
    'lightgbm': {
        'model': LGBMClassifier,
        'params': {
            'n_estimators': 1600,
            'max_depth': 15,
            'learning_rate': 0.045,
            'num_leaves': 80,
            'min_child_samples': 85,
            'subsample': 0.87,
            'colsample_bytree': 0.53,
            'reg_alpha': 0.06,
            'reg_lambda': 7.0,
            'random_state': CONFIG['random_state'],
            'verbose': -1,
            'class_weight': 'balanced'
        }
    },
    'xgboost': {
        'model': XGBClassifier,
        'params': {
            'n_estimators': 800,
            'max_depth': 10,
            'learning_rate': 0.032,
            'subsample': 0.82,
            'colsample_bytree': 0.61,
            'gamma': 1.7,
            'reg_alpha': 0.75,
            'reg_lambda': 0.27,
            'random_state': CONFIG['random_state'],
            'eval_metric': 'auc',
            'scale_pos_weight': 7.3  # Handle class imbalance
        }
    },
    'catboost': {
        'model': CatBoostClassifier,
        'params': {
            'iterations': 700,
            'depth': 8,
            'learning_rate': 0.15,
            'l2_leaf_reg': 3.0,
            'bagging_temperature': 0.025,
            'border_count': 230,
            'random_strength': 0.0001,
            'scale_pos_weight': 1.8,
            'random_state': CONFIG['random_state'],
            'verbose': False
        }
    },
    'random_forest': {
        'model': RandomForestClassifier,
        'params': {
            'n_estimators': 500,
            'max_depth': 20,
            'min_samples_split': 10,
            'min_samples_leaf': 4,
            'class_weight': 'balanced',
            'random_state': CONFIG['random_state'],
            'n_jobs': -1
        }
    }
}

# ================================================================================
# ADVANCED ENSEMBLE CLASS
# ================================================================================
class AdvancedEnsemble:
    """Advanced ensemble with multiple models and intelligent blending"""
    
    def __init__(self, model_configs, n_folds=5):
        self.model_configs = model_configs
        self.n_folds = n_folds
        self.models = {name: [] for name in model_configs.keys()}
        self.oof_predictions = {}
        self.test_predictions = {}
        self.cv_scores = {}
        
    def train_model(self, model_name, X, y, X_test):
        """Train a single model with cross-validation"""
        print(f"\n{'='*50}")
        print(f"TRAINING {model_name.upper()} MODEL")
        print(f"{'='*50}")
        
        config = self.model_configs[model_name]
        oof_preds = np.zeros(len(X))
        test_preds = np.zeros(X_test.shape[0])
        scores = []
        
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=CONFIG['random_state'])
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            print(f"Fold {fold + 1}/{self.n_folds}")
            
            # Split data
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
            
            # Initialize and train model
            model = config['model'](**config['params'])
            model.fit(X_train_fold, y_train_fold)
            
            # Predictions
            val_pred = model.predict_proba(X_val_fold)[:, 1]
            test_pred = model.predict_proba(X_test)[:, 1]
            
            # Store predictions
            oof_preds[val_idx] = val_pred
            test_preds += test_pred
            
            # Calculate score
            score = roc_auc_score(y_val_fold, val_pred)
            scores.append(score)
            
            # Store model
            self.models[model_name].append(model)
            
            print(f"  AUC: {score:.6f}")
            
            # Memory cleanup
            del model, X_train_fold, X_val_fold, y_train_fold, y_val_fold
            gc.collect()
        
        # Average test predictions
        test_preds /= self.n_folds
        
        # Store results
        self.oof_predictions[model_name] = oof_preds
        self.test_predictions[model_name] = test_preds
        self.cv_scores[model_name] = {
            'mean': np.mean(scores),
            'std': np.std(scores),
            'scores': scores
        }
        
        print(f"\nCV Score: {np.mean(scores):.6f} Â± {np.std(scores):.6f}")
        
        return oof_preds, test_preds
    
    def train_all_models(self, X, y, X_test):
        """Train all models in the ensemble"""
        print("\n" + "="*80)
        print("TRAINING ENSEMBLE MODELS")
        print("="*80)
        
        for model_name in self.model_configs.keys():
            self.train_model(model_name, X, y, X_test)
            
    def create_blend_ensemble(self, y):
        """Create optimized blend of all models"""
        print("\n" + "="*50)
        print("CREATING ENSEMBLE BLEND")
        print("="*50)
        
        # Simple weighted average based on CV performance
        weights = {}
        total_weight = 0
        
        for model_name, score_info in self.cv_scores.items():
            # Weight based on CV score (higher score = higher weight)
            weight = score_info['mean'] ** 3  # Cubic weighting for better models
            weights[model_name] = weight
            total_weight += weight
        
        # Normalize weights
        for model_name in weights:
            weights[model_name] /= total_weight
            
        print("Model weights:")
        for model_name, weight in weights.items():
            score = self.cv_scores[model_name]['mean']
            print(f"  {model_name}: {weight:.4f} (CV: {score:.6f})")
        
        # Create ensemble predictions
        ensemble_oof = np.zeros(len(list(self.oof_predictions.values())[0]))
        ensemble_test = np.zeros(len(list(self.test_predictions.values())[0]))
        
        for model_name, weight in weights.items():
            ensemble_oof += weight * self.oof_predictions[model_name]
            ensemble_test += weight * self.test_predictions[model_name]
        
        # Calculate ensemble CV score
        ensemble_score = roc_auc_score(y, ensemble_oof)
        print(f"\nEnsemble CV Score: {ensemble_score:.6f}")
        
        return ensemble_oof, ensemble_test, weights

# ================================================================================
# MAIN EXECUTION PIPELINE
# ================================================================================
def main():
    """Main execution pipeline"""
    start_time = datetime.now()
    
    # ====== LOAD DATA ======
    train, test, sample_submission = load_data()
    
    # ====== FEATURE ENGINEERING ======
    print("\n" + "="*60)
    print("FEATURE ENGINEERING")
    print("="*60)
    
    feature_engineer = AdvancedFeatureEngineer()
    
    # Prepare features (remove target from train data for fitting)
    train_features = train.copy()  # Keep original train data intact
    X = feature_engineer.fit_transform(train_features)  # This will automatically drop 'y' column
    X_test = feature_engineer.transform(test)
    y = train[CONFIG['target_col']]  # Extract target from original train data
    
    print(f"Features after engineering: {X.shape[1]}")
    print(f"Feature names: {X.columns.tolist()[:10]}... (showing first 10)")
    
    # ====== MODEL TRAINING ======
    ensemble = AdvancedEnsemble(MODEL_CONFIGS, n_folds=CONFIG['n_folds'])
    ensemble.train_all_models(X, y, X_test)
    
    # ====== ENSEMBLE CREATION ======
    ensemble_oof, ensemble_test, model_weights = ensemble.create_blend_ensemble(y)
    
    # ====== RESULTS SUMMARY ======
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)
    
    print("Individual Model Scores:")
    for model_name, score_info in ensemble.cv_scores.items():
        print(f"  {model_name:12}: {score_info['mean']:.6f} Â± {score_info['std']:.6f}")
    
    ensemble_score = roc_auc_score(y, ensemble_oof)
    print(f"\nEnsemble Score: {ensemble_score:.6f}")
    
    # ====== SAVE PREDICTIONS ======
    sample_submission[CONFIG['target_col']] = ensemble_test
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"ensemble_prediction_{timestamp}.csv"
    sample_submission.to_csv(filename, index=False)
    
    print(f"\nPredictions saved as: {filename}")
    print(f"Sample predictions:")
    print(sample_submission.head(10))
    
    # ====== EXECUTION TIME ======
    end_time = datetime.now()
    total_time = end_time - start_time
    print(f"\nTotal Training Time: {total_time}")
    
    return ensemble, sample_submission

# ================================================================================
# EXECUTE PIPELINE
# ================================================================================
if __name__ == "__main__":
    ensemble, predictions = main()

