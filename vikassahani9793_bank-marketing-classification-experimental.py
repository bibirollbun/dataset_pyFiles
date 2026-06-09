# ADVANCED BANK MARKETING CLASSIFICATION (UPGRADED AND FIXED)
# Optimized for Tesla T4 Dual GPU (15GB each)

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# GPU, Multi-processing, and Tuning
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import optuna
import multiprocessing as mp

# Machine Learning
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

# Feature Engineering
from sklearn.feature_selection import SelectKBest, f_classif

# Utilities
import gc
import joblib
import os

# Set random seeds for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)
    
# Constants
N_SPLITS = 5
N_TRIALS_OPTUNA = 25 # Number of tuning trials per model
FEATURE_SELECTION_K = 150 # Number of features to select

print("=== ADVANCED BANK MARKETING CLASSIFICATION (UPGRADED AND FIXED) ===")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

# =============================================================================
# DATA LOADING
# =============================================================================
def load_data():
    """Load data from Kaggle or create synthetic data."""
    print("\nğŸ”� Loading data...")
    try:
        train_df = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
        test_df = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')
        sample_sub = pd.read_csv('/kaggle/input/playground-series-s5e8/sample_submission.csv')
    except FileNotFoundError:
        print("Kaggle data not found, creating synthetic data for testing...")
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=10000, n_features=20, n_informative=10, n_redundant=5, random_state=RANDOM_STATE)
        train_df = pd.DataFrame(X, columns=[f'feat_{i}' for i in range(20)])
        train_df['y'] = y
        train_df['id'] = range(len(train_df))
        test_df = pd.DataFrame(X[:5000], columns=[f'feat_{i}' for i in range(20)])
        test_df['id'] = range(len(train_df), len(train_df) + len(test_df))
        sample_sub = pd.DataFrame({'id': test_df['id'], 'y': 0.5})

    print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
    return train_df, test_df, sample_sub

# =============================================================================
# ADVANCED FEATURE ENGINEERING (REFACTORED FOR CV)
# =============================================================================
class AdvancedFeatureEngineer:
    """Refactored feature engineering pipeline for use in cross-validation."""
    def __init__(self):
        self.label_encoders = {}
        self.freq_maps = {}
        self.target_maps = {}
        self.global_target_mean = 0

    def fit(self, df, y=None):
        print("Fitting feature engineer...")
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        self.global_target_mean = y.mean() if y is not None else 0
        
        for col in categorical_cols:
            le = LabelEncoder()
            unique_vals = df[col].astype(str).unique()
            le.fit(unique_vals)
            self.label_encoders[col] = le
            
            self.freq_maps[col] = df[col].value_counts().to_dict()
            if y is not None:
                self.target_maps[col] = y.groupby(df[col]).mean().to_dict()
        return self

    def transform(self, df):
        df = df.copy()
        
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            if col in self.label_encoders:
                # Handle unseen values during transform
                df[f'{col}_encoded'] = df[col].astype(str).map(lambda s: self.label_encoders[col].transform([s])[0] if s in self.label_encoders[col].classes_ else -1)
                df[f'{col}_freq'] = df[col].map(self.freq_maps[col]).fillna(0)
                if self.target_maps and col in self.target_maps:
                    df[f'{col}_target_enc'] = df[col].map(self.target_maps[col]).fillna(self.global_target_mean)
        
        df = df.drop(columns=[col for col in categorical_cols if col in df.columns])
        
        numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
        numerical_cols = [c for c in numerical_cols if c not in ['id', 'y']]

        if len(numerical_cols) > 6:
            for i, col1 in enumerate(numerical_cols[:5]):
                for col2 in numerical_cols[i+1:6]:
                    df[f'{col1}_{col2}_mult'] = df[col1] * df[col2]
                    df[f'{col1}_{col2}_div'] = df[col1] / (df[col2] + 1e-6)
        
        if numerical_cols:
            df['row_mean'] = df[numerical_cols].mean(axis=1)
            df['row_std'] = df[numerical_cols].std(axis=1)
        
        return df

# =============================================================================
# MODELS AND HYPERPARAMETER TUNING
# =============================================================================
class TabularNN(nn.Module):
    def __init__(self, input_dim, hidden_dims=[512, 256], dropout=0.3):
        super(TabularNN, self).__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.BatchNorm1d(prev_dim),
                nn.Dropout(dropout),
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze()

def get_tuned_params(X_train, y_train, X_val, y_val):
    """Run Optuna to find best hyperparameters for all models."""
    print("\nâš™ï¸�  Running hyperparameter tuning with Optuna...")
    
    all_best_params = {}

    # --- LightGBM Tuning ---
    def lgb_objective(trial):
        params = {
            'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
            'device': 'gpu' if torch.cuda.is_available() else 'cpu',
            'n_estimators': 1000, 'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'num_leaves': trial.suggest_int('num_leaves', 20, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'random_state': RANDOM_STATE, 'verbosity': -1
        }
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
        preds = model.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, preds)

    study_lgb = optuna.create_study(direction='maximize')
    study_lgb.optimize(lgb_objective, n_trials=N_TRIALS_OPTUNA)
    all_best_params['lightgbm'] = study_lgb.best_params
    print(f"Best LGBM AUC: {study_lgb.best_value:.6f}")

    # --- XGBoost Tuning ---
    def xgb_objective(trial):
        params = {
            'objective': 'binary:logistic', 'eval_metric': 'auc',
            'tree_method': 'gpu_hist' if torch.cuda.is_available() else 'hist',
            'n_estimators': 1000, 'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'random_state': RANDOM_STATE
        }
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=False)
        preds = model.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, preds)

    study_xgb = optuna.create_study(direction='maximize')
    study_xgb.optimize(xgb_objective, n_trials=N_TRIALS_OPTUNA)
    all_best_params['xgboost'] = study_xgb.best_params
    print(f"Best XGBoost AUC: {study_xgb.best_value:.6f}")
    
    return all_best_params

# =============================================================================
# STACKING ENSEMBLE IMPLEMENTATION
# =============================================================================
class EnsembleStacker:
    """Class to manage training, prediction, and stacking of models."""
    def __init__(self, base_models, meta_model):
        self.base_models = base_models
        self.meta_model = meta_model

    def fit_predict(self, X, y, X_test):
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        
        oof_preds = np.zeros((len(X), len(self.base_models)))
        test_preds = np.zeros((len(X_test), len(self.base_models)))

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            print(f"\n--- Fold {fold + 1}/{N_SPLITS} ---")
            X_train_fold, y_train_fold = X[train_idx], y[train_idx]
            X_val_fold, _ = X[val_idx], y[val_idx]
            
            scaler = RobustScaler()
            X_train_fold = scaler.fit_transform(X_train_fold)
            X_val_fold = scaler.transform(X_val_fold)
            X_test_scaled = scaler.transform(X_test)
            
            for i, (name, model) in enumerate(self.base_models.items()):
                print(f"Training {name}...")
                model.fit(X_train_fold, y_train_fold)
                
                oof_preds[val_idx, i] = model.predict_proba(X_val_fold)[:, 1]
                test_preds[:, i] += model.predict_proba(X_test_scaled)[:, 1] / N_SPLITS
        
        print("\n--- Training Meta-Model ---")
        self.meta_model.fit(oof_preds, y)
        
        final_oof_score = roc_auc_score(y, self.meta_model.predict_proba(oof_preds)[:, 1])
        print(f"Ensemble OOF AUC Score: {final_oof_score:.6f}")
        
        final_test_preds = self.meta_model.predict_proba(test_preds)[:, 1]
        return final_test_preds

# =============================================================================
# MAIN EXECUTION PIPELINE
# =============================================================================
def main():
    """Main execution pipeline."""
    train_df, test_df, sample_sub = load_data()
    
    y = train_df['y']
    train_ids, test_ids = train_df['id'], test_df['id']
    train_df = train_df.drop(columns=['id', 'y'])
    test_df = test_df.drop(columns=['id'])

    feature_engineer = AdvancedFeatureEngineer()
    feature_engineer.fit(train_df, y)
    train_processed = feature_engineer.transform(train_df)
    test_processed = feature_engineer.transform(test_df)
    
    train_cols = train_processed.columns
    test_cols = test_processed.columns
    shared_cols = list(set(train_cols) & set(test_cols))
    train_processed = train_processed[shared_cols]
    test_processed = test_processed[shared_cols]
    
    print(f"Shapes after FE: Train {train_processed.shape}, Test {test_processed.shape}")
    
    X = train_processed.values
    X_test = test_processed.values

    print("\nğŸ¤º Performing feature selection...")
    selector = SelectKBest(f_classif, k=min(FEATURE_SELECTION_K, X.shape[1]))
    X = selector.fit_transform(X, y)
    X_test = selector.transform(X_test)
    print(f"Shape after feature selection: {X.shape}")

    del train_df, test_df, train_processed, test_processed
    gc.collect()

    # --- Hyperparameter Tuning ---
    # Use a subset of data for faster tuning
    
    # ============================ FIXED BLOCK ============================
    # Correctly get the two index arrays first
    train_idx, val_idx = next(StratifiedKFold(n_splits=4, shuffle=True, random_state=RANDOM_STATE).split(X, y))

    # Now, use the indices to create your four data subsets
    X_train_tune, X_val_tune = X[train_idx], X[val_idx]
    y_train_tune, y_val_tune = y.iloc[train_idx].values, y.iloc[val_idx].values
    # =====================================================================
    
    scaler_tune = RobustScaler()
    X_train_tune_scaled = scaler_tune.fit_transform(X_train_tune)
    X_val_tune_scaled = scaler_tune.transform(X_val_tune)
    
    best_params = get_tuned_params(X_train_tune_scaled, y_train_tune, X_val_tune_scaled, y_val_tune)

    print("\nğŸš€ Starting ensemble training with cross-validation...")
    
    base_models = {
        'lightgbm': lgb.LGBMClassifier(
            **best_params['lightgbm'], random_state=RANDOM_STATE, n_estimators=2000,
            device='gpu' if torch.cuda.is_available() else 'cpu'
        ),
        'xgboost': xgb.XGBClassifier(
            **best_params['xgboost'], random_state=RANDOM_STATE, n_estimators=2000,
            tree_method='gpu_hist' if torch.cuda.is_available() else 'hist', use_label_encoder=False
        ),
        'catboost': cb.CatBoostClassifier(
            iterations=2000, learning_rate=0.03, depth=8,
            task_type='GPU' if torch.cuda.is_available() else 'CPU',
            random_seed=RANDOM_STATE, verbose=0, early_stopping_rounds=50
        )
    }

    meta_model = LogisticRegression(C=1)

    stacker = EnsembleStacker(base_models, meta_model)
    test_predictions = stacker.fit_predict(X, y.values, X_test)

    print("\nğŸ”® Generating submission file...")
    submission = sample_sub.copy()
    submission['y'] = test_predictions
    submission.to_csv('stacked_ensemble_submission.csv', index=False)
    print("Submission saved as 'stacked_ensemble_submission.csv'")
    
    print("\nâœ… Pipeline completed!")


if __name__ == "__main__":
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
        
    main()

