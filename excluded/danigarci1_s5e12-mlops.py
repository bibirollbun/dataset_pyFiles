#config.py

import os
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class PathConfig:
    """File path configurations"""
    # Input paths (adjust for your environment)
    TRAIN_DATA_PATH: str = "/kaggle/input/playground-series-s5e12/train.csv"
    TEST_DATA_PATH: str = "/kaggle/input/playground-series-s5e12/test.csv"
    
    # Output paths
    OUTPUT_DIR: str = "/kaggle/working/"
    MODEL_DIR: str = "/kaggle/working/models/"
    SUBMISSION_FILE: str = "submission.csv"
    
    # Visualization outputs
    MODEL_COMPARISON_PLOT: str = "model_comparison.png"
    SHAP_SUMMARY_PLOT: str = "shap_summary_best_model.png"
    FEATURE_IMPORTANCE_PLOT: str = "feature_importance_best_model.png"


@dataclass
class ModelConfig:
    """Model training configurations"""
    # Target and ID columns
    TARGET_COL: str = "diagnosed_diabetes"
    ID_COL: str = "id"
    
    # Models to test
    MODELS_TO_TEST: List[str] = None
    
    # GPU settings
    USE_GPU: bool = False
    
    # Random seed for reproducibility
    RANDOM_SEED: int = 42
    
    # Optuna optimization settings
    N_FOLDS_OPT: int = 3  # Fewer folds for hyperparameter optimization
    N_TRIALS: int = 15  # Number of Optuna trials
    MAX_OPT_SAMPLES: int = 100000  # Sample size for Optuna tuning
    
    # Final training settings
    N_FOLDS_SUB: int = 5  # More folds for final model training
    
    # Early stopping
    EARLY_STOPPING_ROUNDS: int = 50
    
    # Model iterations
    MAX_ITERATIONS: int = 1500
    
    def __post_init__(self):
        if self.MODELS_TO_TEST is None:
            self.MODELS_TO_TEST = ['xgboost', 'lightgbm', 'catboost']


@dataclass
class FeatureConfig:
    """Feature engineering configurations"""
    # Columns to convert from numerical to categorical
    NUMERICAL_TO_BOOLEAN: List[str] = None
    
    # Columns to apply log transformation
    SKEWED_TO_GAUSS: List[str] = None
    
    def __post_init__(self):
        if self.NUMERICAL_TO_BOOLEAN is None:
            self.NUMERICAL_TO_BOOLEAN = [
                "cardiovascular_history",
                "hypertension_history",
                "family_history_diabetes"
            ]
        
        if self.SKEWED_TO_GAUSS is None:
            self.SKEWED_TO_GAUSS = ['physical_activity_minutes_per_week']


class Config:
    """Main configuration class that combines all configs"""
    def __init__(self):
        self.paths = PathConfig()
        self.model = ModelConfig()
        self.features = FeatureConfig()
        
        # Create necessary directories
        self._create_directories()
    
    def _create_directories(self):
        """Create output directories if they don't exist"""
        os.makedirs(self.paths.MODEL_DIR, exist_ok=True)
        os.makedirs(self.paths.OUTPUT_DIR, exist_ok=True)
    
    def get_hyperparameter_space(self, model_name: str, trial) -> Dict[str, Any]:
        """
        Define hyperparameter search space for each model
        
        Args:
            model_name: Name of the model
            trial: Optuna trial object
            
        Returns:
            Dictionary of hyperparameters
        """
        if model_name == 'xgboost':
            return {
                'n_estimators': self.model.MAX_ITERATIONS,
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'objective': 'binary:logistic',
                'tree_method': 'gpu_hist' if self.model.USE_GPU else 'hist',
                'enable_categorical': True,
                'n_jobs': -1,
                'random_state': self.model.RANDOM_SEED,
                'eval_metric': 'auc',
                'early_stopping_rounds': self.model.EARLY_STOPPING_ROUNDS
            }
        
        elif model_name == 'lightgbm':
            return {
                'n_estimators': self.model.MAX_ITERATIONS,
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 20, 200),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'objective': 'binary',
                'device': 'gpu' if self.model.USE_GPU else 'cpu',
                'n_jobs': -1,
                'verbosity': -1,
                'random_state': self.model.RANDOM_SEED
            }
        
        elif model_name == 'catboost':
            return {
                'iterations': self.model.MAX_ITERATIONS,
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'depth': trial.suggest_int('depth', 4, 10),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-2, 10, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'loss_function': 'Logloss',
                'task_type': 'GPU' if self.model.USE_GPU else 'CPU',
                'thread_count': -1,
                'random_seed': self.model.RANDOM_SEED,
            }
        
        else:
            raise ValueError(f"Unknown model: {model_name}")


#utils.py
import pandas as pd
import numpy as np
import json
import pickle
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def save_json(data: Dict, filepath: str):
    """
    Save dictionary to JSON file
    
    Args:
        data: Dictionary to save
        filepath: Path to save file
    """
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)
    logger.info(f"Saved JSON to {filepath}")


def load_json(filepath: str) -> Dict:
    """
    Load JSON file
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        Dictionary with JSON contents
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    logger.info(f"Loaded JSON from {filepath}")
    return data


def save_pickle(obj: Any, filepath: str):
    """
    Save object using pickle
    
    Args:
        obj: Object to save
        filepath: Path to save file
    """
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)
    logger.info(f"Saved pickle to {filepath}")


def load_pickle(filepath: str) -> Any:
    """
    Load pickled object
    
    Args:
        filepath: Path to pickle file
        
    Returns:
        Loaded object
    """
    with open(filepath, 'rb') as f:
        obj = pickle.load(f)
    logger.info(f"Loaded pickle from {filepath}")
    return obj


def create_timestamp() -> str:
    """
    Create timestamp string for versioning
    
    Returns:
        Timestamp string in format YYYYMMDD_HHMMSS
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(directory: str):
    """
    Create directory if it doesn't exist
    
    Args:
        directory: Path to directory
    """
    os.makedirs(directory, exist_ok=True)


def get_file_size(filepath: str) -> str:
    """
    Get human-readable file size
    
    Args:
        filepath: Path to file
        
    Returns:
        File size string (e.g., "1.5 MB")
    """
    size_bytes = os.path.getsize(filepath)
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def log_dataframe_info(df: pd.DataFrame, name: str = "DataFrame"):
    """
    Log comprehensive dataframe information
    
    Args:
        df: Dataframe to analyze
        name: Name for logging
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"{name} Information")
    logger.info(f"{'='*60}")
    logger.info(f"Shape: {df.shape}")
    logger.info(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    logger.info(f"\nColumn types:")
    logger.info(f"{df.dtypes.value_counts()}")
    logger.info(f"\nMissing values:")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        logger.info(f"{missing[missing > 0]}")
    else:
        logger.info("No missing values")
    logger.info(f"{'='*60}\n")


def calculate_class_weights(y: pd.Series) -> Dict[int, float]:
    """
    Calculate class weights for imbalanced datasets
    
    Args:
        y: Target variable
        
    Returns:
        Dictionary mapping class to weight
    """
    from sklearn.utils.class_weight import compute_class_weight
    
    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    
    class_weights = {classes[i]: weights[i] for i in range(len(classes))}
    logger.info(f"Class weights: {class_weights}")
    
    return class_weights


def reduce_mem_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Reduce memory usage of dataframe by downcasting numeric types
    
    Args:
        df: Dataframe to optimize
        verbose: Whether to print memory reduction info
        
    Returns:
        Optimized dataframe
    """
    start_mem = df.memory_usage().sum() / 1024**2
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object and col_type.name != 'category':
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float32)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    
    end_mem = df.memory_usage().sum() / 1024**2
    
    if verbose:
        logger.info(f'Memory usage decreased from {start_mem:.2f} MB to {end_mem:.2f} MB '
                   f'({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)')
    
    return df


def validate_submission(
    submission: pd.DataFrame,
    test_df: pd.DataFrame,
    id_col: str,
    target_col: str
) -> List[str]:
    """
    Validate submission file for common issues
    
    Args:
        submission: Submission dataframe
        test_df: Original test dataframe
        id_col: ID column name
        target_col: Target column name
        
    Returns:
        List of validation errors (empty if no errors)
    """
    errors = []
    
    # Check required columns
    if id_col not in submission.columns:
        errors.append(f"Missing ID column: {id_col}")
    if target_col not in submission.columns:
        errors.append(f"Missing target column: {target_col}")
    
    # Check for missing values
    if submission[target_col].isnull().any():
        errors.append("Submission contains null values")
    
    # Check ID alignment
    if not submission[id_col].equals(test_df[id_col]):
        errors.append("ID column does not match test data")
    
    # Check prediction range (for probabilities)
    if (submission[target_col] < 0).any() or (submission[target_col] > 1).any():
        errors.append("Predictions outside valid range [0, 1]")
    
    # Check for duplicates
    if submission[id_col].duplicated().any():
        errors.append("Duplicate IDs found in submission")
    
    if len(errors) == 0:
        logger.info("✓ Submission validation passed")
    else:
        logger.warning(f"✗ Submission validation failed with {len(errors)} errors")
        for error in errors:
            logger.warning(f"  - {error}")
    
    return errors


def print_training_summary(results: Dict):
    """
    Print formatted training summary
    
    Args:
        results: Results dictionary from training pipeline
    """
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)
    
    print(f"\nBest Model: {results['best_model_name'].upper()}")
    print(f"CV AUC: {results['study_results'][results['best_model_name']]['best_score']:.4f}")
    
    print("\n" + "-"*80)
    print("All Models Performance:")
    print("-"*80)
    for model_name, result in results['study_results'].items():
        print(f"  {model_name.upper():15s}: {result['best_score']:.4f}")
    
    print("\n" + "-"*80)
    print("Top 10 Most Important Features:")
    print("-"*80)
    fi_df = pd.DataFrame({
        'feature': results['feature_names'],
        'importance': results['feature_importances']
    })
    fi_df = fi_df.sort_values('importance', ascending=False).head(10)
    for idx, row in fi_df.iterrows():
        print(f"  {row['feature']:40s}: {row['importance']:.4f}")
    
    print("\n" + "="*80)


def merge_fold_predictions(
    fold_predictions: List[np.ndarray],
    method: str = 'mean'
) -> np.ndarray:
    """
    Merge predictions from multiple folds
    
    Args:
        fold_predictions: List of prediction arrays
        method: Merging method ('mean', 'median', 'geometric_mean')
        
    Returns:
        Merged predictions
    """
    if method == 'mean':
        return np.mean(fold_predictions, axis=0)
    elif method == 'median':
        return np.median(fold_predictions, axis=0)
    elif method == 'geometric_mean':
        return np.exp(np.mean(np.log(fold_predictions), axis=0))
    else:
        raise ValueError(f"Unknown merge method: {method}")


import os
import joblib
import pandas as pd
import numpy as np
from typing import Tuple, List, Optional, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """MLOps-capable data preprocessor.

    Key features:
    - `fit` on training dataframe (records category lists, feature names, indices)
    - `transform` to apply same transforms to other frames without leaking
      information from test data
    - `save` / `load` for reuse in inference pipelines
    """

    def __init__(self, config):
        self.config = config
        self.fitted = False
        # learned state
        self.category_map: Dict[str, List] = {}
        self.cat_features_indices: List[int] = []
        self.feature_names: List[str] = []
        self._numerical_to_boolean = getattr(self.config.features, 'NUMERICAL_TO_BOOLEAN', [])
        self._skewed_cols = getattr(self.config.features, 'SKEWED_TO_GAUSS', [])

    # ---------- I/O helpers ----------
    def load_data(self, path: str) -> pd.DataFrame:
        logger.info(f"Loading data from {path}")
        return pd.read_csv(path)

    def save(self, path: str) -> None:
        """Save the fitted preprocessor to `path` using joblib."""
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        joblib.dump(self, path)
        logger.info(f"Saved preprocessor to {path}")

    @classmethod
    def load(cls, path: str) -> 'DataPreprocessor':
        """Load a saved preprocessor artifact."""
        logger.info(f"Loading preprocessor artifact from {path}")
        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError("Loaded object is not a DataPreprocessor instance")
        return obj

    # ---------- Fit / Transform API ----------
    def fit(self, train_df: pd.DataFrame) -> None:
        """Fit the preprocessor on training dataframe.

        This records:
        - categorical value lists per categorical column
        - feature names and categorical indices
        """
        # Work on a copy
        df = train_df.copy()

        # Convert numerical-to-boolean columns to categorical dtype
        if self._numerical_to_boolean:
            for col in self._numerical_to_boolean:
                if col in df.columns:
                    df[col] = df[col].astype('category')

        # Apply log transform to skewed columns (train only)
        if self._skewed_cols:
            for col in self._skewed_cols:
                if col in df.columns:
                    # Guard against negative values; clip at -1e-6 then log1p
                    df[col] = np.log1p(np.clip(df[col].astype(float), a_min=0.0, a_max=None))

        # Identify object columns and record their categories from training set
        # Use a sentinel for missing values so CatBoost sees strings instead of NaN
        MISSING_SENTINEL = "__MISSING__"
        object_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        for col in object_cols:
            # Replace NaN with sentinel and convert to string for category discovery.
            # Cast to object first to avoid errors when the column is already
            # a pandas Categorical without the sentinel present.
            series_for_cats = df[col].astype('object').fillna(MISSING_SENTINEL).astype('str')
            cats = pd.Series(series_for_cats).unique().tolist()
            self.category_map[col] = cats
            # Apply categorical dtype with learned categories and ensure missing replaced
            # Cast to object first to avoid assigning new categories into an
            # existing Categorical (which raises TypeError). Then re-cast to
            # the categorical dtype containing the sentinel.
            df[col] = (
                df[col]
                .astype('object')
                .fillna(MISSING_SENTINEL)
                .astype('str')
                .astype(pd.CategoricalDtype(categories=cats, ordered=False))
            )

        # Build feature matrix
        if self.config.model.TARGET_COL not in df.columns:
            raise KeyError(f"Target column {self.config.model.TARGET_COL} not found in training data")

        X = df.drop(columns=[self.config.model.TARGET_COL, self.config.model.ID_COL])
        y = df[self.config.model.TARGET_COL]

        self.feature_names = X.columns.tolist()
        self.cat_features_indices = [i for i, c in enumerate(self.feature_names) if X[c].dtype.name == 'category']
        self.fitted = True

        logger.info(f"Fitted preprocessor on data shape: {df.shape}")
        logger.info(f"Found {len(self.category_map)} categorical columns")

        # keep y if needed by user; don't store train data
        return None

    def fit_transform(self, train_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Fit on `train_df` and return transformed X and y."""
        self.fit(train_df)

        df = train_df.copy()

        # Same operations as in fit
        if self._numerical_to_boolean:
            for col in self._numerical_to_boolean:
                if col in df.columns:
                    df[col] = df[col].astype('category')

        if self._skewed_cols:
            for col in self._skewed_cols:
                if col in df.columns:
                    df[col] = np.log1p(np.clip(df[col].astype(float), a_min=0.0, a_max=None))

        # Apply categorical dtypes using learned categories; replace missing with sentinel
        MISSING_SENTINEL = "__MISSING__"
        for col, cats in self.category_map.items():
            if col in df.columns:
                # Ensure we cast to object first to allow introducing the
                # missing sentinel category when needed.
                df[col] = (
                    df[col]
                    .astype('object')
                    .fillna(MISSING_SENTINEL)
                    .astype('str')
                    .astype(pd.CategoricalDtype(categories=cats, ordered=False))
                )

        X = df.drop(columns=[self.config.model.TARGET_COL, self.config.model.ID_COL])
        y = df[self.config.model.TARGET_COL]

        # update cat indices in case types changed
        self.feature_names = X.columns.tolist()
        self.cat_features_indices = [i for i, c in enumerate(self.feature_names) if X[c].dtype.name == 'category']

        return X, y

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply learned transformations to a dataframe (no label column expected).

        Raises if preprocessor is not fitted.
        """
        if not self.fitted:
            raise RuntimeError("Preprocessor must be fitted or loaded before calling transform()")

        out = df.copy()

        # Convert numerical-to-boolean columns to categorical dtype
        if self._numerical_to_boolean:
            for col in self._numerical_to_boolean:
                if col in out.columns:
                    out[col] = out[col].astype('category')

        # Apply log transform using same columns
        if self._skewed_cols:
            for col in self._skewed_cols:
                if col in out.columns:
                    out[col] = np.log1p(np.clip(out[col].astype(float), a_min=0.0, a_max=None))

        # Apply categorical dtypes using categories learned from training and set missing sentinel
        MISSING_SENTINEL = "__MISSING__"
        for col, cats in self.category_map.items():
            if col in out.columns:
                # Cast to object first to avoid errors when inserting the
                # missing sentinel into existing categorical columns.
                out[col] = (
                    out[col]
                    .astype('object')
                    .fillna(MISSING_SENTINEL)
                    .astype('str')
                    .astype(pd.CategoricalDtype(categories=cats, ordered=False))
                )

        # Ensure we return the same feature columns and order the same as training
        missing = [c for c in self.feature_names if c not in out.columns]
        if missing:
            raise KeyError(f"Missing expected feature columns: {missing}")

        out = out[self.feature_names]

        # update cat_features_indices just in case
        self.cat_features_indices = [i for i, c in enumerate(self.feature_names) if out[c].dtype.name == 'category']

        return out

    def transform_for_inference(self, test_df: pd.DataFrame) -> pd.DataFrame:
        """Transform incoming test/inference data and drop ID column before returning X_test."""
        out = self.transform(test_df)
        # Drop ID column if present in original test_df
        if self.config.model.ID_COL in out.columns:
            out = out.drop(columns=[self.config.model.ID_COL])
        return out


def create_preprocessor(config):
    return DataPreprocessor(config)




import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List, Optional
import logging
import warnings
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
import optuna
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Handles model training and hyperparameter optimization"""
    
    def __init__(self, config, cat_features_indices: Optional[List[int]] = None):
        """
        Initialize trainer with configuration
        
        Args:
            config: Configuration object
            cat_features_indices: List of categorical feature indices
        """
        self.config = config
        self.cat_features_indices = cat_features_indices or []
        self.study_results = {}
        self.best_model_name = None
        self.best_params = None
        
    def get_model_class(self, model_name: str):
        """
        Get model class based on name
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model class
        """
        if model_name == 'xgboost':
            return xgb.XGBClassifier
        elif model_name == 'lightgbm':
            return lgb.LGBMClassifier
        elif model_name == 'catboost':
            return cb.CatBoostClassifier
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    def run_cv_with_early_stopping(
        self, 
        model_name: str, 
        params: Dict[str, Any], 
        X: pd.DataFrame, 
        y: pd.Series, 
        n_folds: int
    ) -> float:
        """
        Run cross-validation with early stopping for faster training
        
        Args:
            model_name: Name of the model
            params: Model hyperparameters
            X: Feature matrix
            y: Target vector
            n_folds: Number of CV folds
            
        Returns:
            Mean AUC score across folds
        """
        kf = StratifiedKFold(
            n_splits=n_folds, 
            shuffle=True, 
            random_state=self.config.model.RANDOM_SEED
        )
        fold_scores = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
            
            # Initialize model
            if model_name == 'xgboost':
                model = xgb.XGBClassifier(**params)
                model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
                
            elif model_name == 'lightgbm':
                model = lgb.LGBMClassifier(**params)
                callbacks = [
                    lgb.early_stopping(
                        stopping_rounds=self.config.model.EARLY_STOPPING_ROUNDS, 
                        verbose=False
                    )
                ]
                model.fit(
                    X_tr, y_tr, 
                    eval_set=[(X_val, y_val)], 
                    eval_metric='auc', 
                    callbacks=callbacks
                )
                
            elif model_name == 'catboost':
                model = cb.CatBoostClassifier(**params)
                model.fit(
                    X_tr, y_tr, 
                    eval_set=(X_val, y_val), 
                    early_stopping_rounds=self.config.model.EARLY_STOPPING_ROUNDS, 
                    verbose=False
                )
            
            # Predict and score
            preds = model.predict_proba(X_val)[:, 1]
            
            try:
                score = roc_auc_score(y_val, preds)
                fold_scores.append(score)
            except Exception as e:
                logger.warning(f"Could not calculate AUC for fold {fold_idx}: {e}")
                fold_scores.append(0.5)  # Fallback score
        
        return np.mean(fold_scores)
    
    def optimize_hyperparameters(
        self, 
        model_name: str, 
        X: pd.DataFrame, 
        y: pd.Series
    ) -> Dict[str, Any]:
        """
        Optimize hyperparameters using Optuna
        
        Args:
            model_name: Name of the model
            X: Feature matrix
            y: Target vector
            
        Returns:
            Dictionary with best score and best parameters
        """
        logger.info(f"Optimizing hyperparameters for {model_name.upper()}")
        
        # Sample data if too large
        if len(X) > self.config.model.MAX_OPT_SAMPLES:
            logger.info(
                f"Dataset size ({len(X)}) exceeds limit. "
                f"Subsampling to {self.config.model.MAX_OPT_SAMPLES} rows."
            )
            X_opt, _, y_opt, _ = train_test_split(
                X, y, 
                train_size=self.config.model.MAX_OPT_SAMPLES, 
                stratify=y, 
                random_state=self.config.model.RANDOM_SEED
            )
        else:
            X_opt, y_opt = X, y
        
        # Define objective function
        def objective(trial):
            params = self.config.get_hyperparameter_space(model_name, trial)
            
            # Add cat_features for CatBoost
            if model_name == 'catboost' and self.cat_features_indices:
                params['cat_features'] = self.cat_features_indices
            
            avg_score = self.run_cv_with_early_stopping(
                model_name, params, X_opt, y_opt, 
                n_folds=self.config.model.N_FOLDS_OPT
            )
            return avg_score
        
        # Run optimization
        sampler = optuna.samplers.TPESampler(seed=self.config.model.RANDOM_SEED)
        study = optuna.create_study(direction='maximize', sampler=sampler)
        
        # Suppress Optuna logs
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        study.optimize(objective, n_trials=self.config.model.N_TRIALS, show_progress_bar=True)
        
        logger.info(f"Best AUC for {model_name}: {study.best_value:.4f}")
        
        return {
            'best_score': study.best_value,
            'best_params': study.best_params,
            'study': study
        }
    
    def optimize_all_models(
        self, 
        X: pd.DataFrame, 
        y: pd.Series
    ) -> Dict[str, Dict[str, Any]]:
        """
        Optimize hyperparameters for all configured models
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            Dictionary with results for each model
        """
        logger.info("=" * 60)
        logger.info("Starting hyperparameter optimization for all models")
        logger.info("=" * 60)
        
        for model_name in self.config.model.MODELS_TO_TEST:
            result = self.optimize_hyperparameters(model_name, X, y)
            self.study_results[model_name] = result
        
        # Identify best model
        self.best_model_name = max(
            self.study_results, 
            key=lambda k: self.study_results[k]['best_score']
        )
        self.best_params = self.study_results[self.best_model_name]['best_params']
        
        logger.info("=" * 60)
        logger.info(f"Best model: {self.best_model_name.upper()}")
        logger.info(f"Best CV AUC: {self.study_results[self.best_model_name]['best_score']:.4f}")
        logger.info("=" * 60)
        
        return self.study_results
    
    def train_final_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_name: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> Tuple[Optional[np.ndarray], List, np.ndarray]:
        """
        Train final model using K-Fold cross-validation.

        NOTE: This method no longer accepts or predicts on test data. Training
        and inference are separated in the MLOps flow — inference should be
        performed with the `InferencePipeline` using a saved preprocessor and
        the trained models.

        Args:
            X: Training features
            y: Training target
            model_name: Model to train (uses best if not specified)
            params: Model parameters (uses best if not specified)

        Returns:
            Tuple of (predictions (always None), fold_models, feature_importances)
        """
        # Use best model if not specified
        if model_name is None:
            model_name = self.best_model_name
        if params is None:
            params = self.best_params.copy()
        
        logger.info("=" * 60)
        logger.info(f"Training final model: {model_name.upper()}")
        logger.info("=" * 60)
        
        # Update parameters with model-specific settings
        if model_name == 'xgboost':
            params.update({
                'n_estimators': self.config.model.MAX_ITERATIONS,
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'tree_method': 'gpu_hist' if self.config.model.USE_GPU else 'hist',
                'enable_categorical': True,
                'early_stopping_rounds': self.config.model.EARLY_STOPPING_ROUNDS
            })
        elif model_name == 'lightgbm':
            params.update({
                'n_estimators': self.config.model.MAX_ITERATIONS,
                'objective': 'binary',
                'device': 'gpu' if self.config.model.USE_GPU else 'cpu',
                'verbosity': -1
            })
        elif model_name == 'catboost':
            params.update({
                'iterations': self.config.model.MAX_ITERATIONS,
                'loss_function': 'Logloss',
                'task_type': 'GPU' if self.config.model.USE_GPU else 'CPU',
                'cat_features': self.cat_features_indices
            })
        
        # Cross-validation
        kf = StratifiedKFold(
            n_splits=self.config.model.N_FOLDS_SUB,
            shuffle=True,
            random_state=self.config.model.RANDOM_SEED
        )
        
        fold_models = []
        fold_preds = []
        feature_importances = np.zeros(len(X.columns))
        ModelClass = self.get_model_class(model_name)
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            logger.info(f"Training Fold {fold+1}/{self.config.model.N_FOLDS_SUB}...")
            
            X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
            
            model = ModelClass(**params)
            
            # Train with early stopping
            if model_name == 'lightgbm':
                model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_val, y_val)],
                    eval_metric='auc',
                    callbacks=[
                        lgb.early_stopping(
                            stopping_rounds=self.config.model.EARLY_STOPPING_ROUNDS,
                            verbose=False
                        )
                    ]
                )
            elif model_name == 'catboost':
                model.fit(
                    X_tr, y_tr,
                    eval_set=(X_val, y_val),
                    early_stopping_rounds=self.config.model.EARLY_STOPPING_ROUNDS,
                    verbose=False
                )
            elif model_name == 'xgboost':
                model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            
            # Store model
            fold_models.append(model)            
            # Accumulate feature importances
            try:
                feature_importances += model.feature_importances_ / self.config.model.N_FOLDS_SUB
            except AttributeError:
                pass
        
        avg_preds = None
        
        logger.info("Final model training completed")
        
        return avg_preds, fold_models, feature_importances


def create_trainer(config, cat_features_indices: Optional[List[int]] = None):
    """
    Factory function to create model trainer
    
    Args:
        config: Configuration object
        cat_features_indices: List of categorical feature indices
        
    Returns:
        ModelTrainer instance
    """
    return ModelTrainer(config, cat_features_indices)



#model_evaluation.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional
import logging
import shap

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Handles model evaluation and visualization"""
    
    def __init__(self, config):
        """
        Initialize evaluator with configuration
        
        Args:
            config: Configuration object
        """
        self.config = config
        
    def plot_model_comparison(
        self, 
        study_results: Dict[str, Dict[str, Any]], 
        save_path: Optional[str] = None
    ):
        """
        Create bar plot comparing model performance
        
        Args:
            study_results: Dictionary with results from all models
            save_path: Path to save the plot
        """
        logger.info("Generating model comparison plot")
        
        model_names = list(study_results.keys())
        scores = [study_results[m]['best_score'] for m in model_names]
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x=model_names, y=scores, palette='viridis')
        plt.title('Model Comparison: Best CV AUC Score', fontsize=15)
        plt.ylabel('AUC Score')
        plt.xlabel('Model')
        plt.ylim(min(scores) - 0.05, max(scores) + 0.05)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add score labels on bars
        for i, v in enumerate(scores):
            plt.text(i, v, f'{v:.4f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = f"{self.config.paths.OUTPUT_DIR}{self.config.paths.MODEL_COMPARISON_PLOT}"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Model comparison plot saved to {save_path}")
        plt.close()
    
    def plot_feature_importance(
        self, 
        feature_names: List[str], 
        feature_importances: np.ndarray,
        model_name: str,
        top_n: int = 20,
        save_path: Optional[str] = None
    ):
        """
        Plot feature importance
        
        Args:
            feature_names: List of feature names
            feature_importances: Array of feature importance scores
            model_name: Name of the model
            top_n: Number of top features to plot
            save_path: Path to save the plot
        """
        logger.info(f"Plotting top {top_n} feature importances")
        
        # Create dataframe and sort
        fi_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importances
        })
        fi_df = fi_df.sort_values(by='importance', ascending=False).head(top_n)
        
        # Plot
        plt.figure(figsize=(10, 8))
        sns.barplot(x='importance', y='feature', data=fi_df, palette='viridis')
        plt.title(f'Top {top_n} Feature Importance - {model_name.upper()}', fontsize=14)
        plt.xlabel('Importance Score')
        plt.ylabel('Feature')
        plt.tight_layout()
        
        if save_path is None:
            save_path = f"{self.config.paths.OUTPUT_DIR}{self.config.paths.FEATURE_IMPORTANCE_PLOT}"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Feature importance plot saved to {save_path}")
        plt.close()
    
    def plot_shap_summary(
        self,
        model,
        X_val: pd.DataFrame,
        model_name: str,
        save_path: Optional[str] = None
    ):
        """
        Generate SHAP summary plot
        
        Args:
            model: Trained model
            X_val: Validation features
            model_name: Name of the model
            save_path: Path to save the plot
        """
        logger.info("Calculating SHAP values")
        
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_val)
            
            # Handle different SHAP value formats
            if isinstance(shap_values, list):
                shap_matrix = shap_values[1]  # For binary classification
            else:
                shap_matrix = shap_values
            
            # Create plot
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_matrix, X_val, show=False)
            plt.title(f'SHAP Summary (Directional Importance) - {model_name.upper()}')
            plt.tight_layout()
            
            if save_path is None:
                save_path = f"{self.config.paths.OUTPUT_DIR}{self.config.paths.SHAP_SUMMARY_PLOT}"
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"SHAP summary plot saved to {save_path}")
            plt.close()
            
        except Exception as e:
            logger.warning(f"Could not generate SHAP plot: {e}")
    
    def evaluate_and_visualize(
        self,
        study_results: Dict[str, Dict[str, Any]],
        final_model,
        X_val: pd.DataFrame,
        feature_names: List[str],
        feature_importances: np.ndarray,
        model_name: str
    ):
        """
        Complete evaluation pipeline with all visualizations
        
        Args:
            study_results: Results from hyperparameter optimization
            final_model: Trained final model
            X_val: Validation features for SHAP
            feature_names: List of feature names
            feature_importances: Feature importance scores
            model_name: Name of the model
        """
        logger.info("=" * 60)
        logger.info("Starting evaluation and visualization")
        logger.info("=" * 60)
        
        # Plot model comparison
        self.plot_model_comparison(study_results)
        
        # Plot feature importance
        self.plot_feature_importance(
            feature_names,
            feature_importances,
            model_name
        )
        
        # Plot SHAP summary
        self.plot_shap_summary(final_model, X_val, model_name)
        
        logger.info("Evaluation and visualization completed")


def create_evaluator(config):
    """
    Factory function to create evaluator
    
    Args:
        config: Configuration object
        
    Returns:
        ModelEvaluator instance
    """
    return ModelEvaluator(config)


import numpy as np
import pandas as pd
from typing import List, Optional, Union
import logging
import pickle
import os

#from data_preprocessing import DataPreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InferencePipeline:
    """Handles model inference and prediction generation"""
    
    def __init__(self, config):
        """
        Initialize inference pipeline
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.models = []
        self.preprocessor = None
        
    def load_models(self, model_dir: str) -> List:
        """
        Load trained models from directory
        
        Args:
            model_dir: Directory containing saved models
            
        Returns:
            List of loaded models
        """
        logger.info(f"Loading models from {model_dir}")
        
        models = []
        model_files = sorted([f for f in os.listdir(model_dir) if f.endswith('.pkl')])
        
        for model_file in model_files:
            path = os.path.join(model_dir, model_file)
            with open(path, 'rb') as f:
                model = pickle.load(f)
                models.append(model)
            logger.info(f"Loaded {model_file}")
        
        self.models = models
        return models
    
    def predict_single_model(
        self, 
        model, 
        X: pd.DataFrame, 
        return_proba: bool = True
    ) -> np.ndarray:
        """
        Make predictions with a single model
        
        Args:
            model: Trained model
            X: Feature matrix
            return_proba: Whether to return probabilities
            
        Returns:
            Predictions array
        """
        if return_proba:
            return model.predict_proba(X)[:, 1]
        else:
            return model.predict(X)
    
    def predict_ensemble(
        self, 
        X: pd.DataFrame, 
        return_proba: bool = True
    ) -> np.ndarray:
        """
        Make predictions using ensemble of models (averaging)
        
        Args:
            X: Feature matrix
            return_proba: Whether to return probabilities
            
        Returns:
            Ensemble predictions
        """
        if not self.models:
            raise ValueError("No models loaded. Call load_models() first.")
        
        logger.info(f"Making predictions with {len(self.models)} models")
        
        predictions = []
        for model in self.models:
            preds = self.predict_single_model(model, X, return_proba)
            predictions.append(preds)
        
        # Average predictions
        ensemble_preds = np.mean(predictions, axis=0)
        
        return ensemble_preds
    
    def create_submission(
        self,
        test_df: pd.DataFrame,
        predictions: np.ndarray,
        id_col: Optional[str] = None,
        target_col: Optional[str] = None,
        output_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Create submission file
        
        Args:
            test_df: Test dataframe (must contain ID column)
            predictions: Prediction values
            id_col: Name of ID column
            target_col: Name of target column
            output_path: Path to save submission file
            
        Returns:
            Submission dataframe
        """
        if id_col is None:
            id_col = self.config.model.ID_COL
        if target_col is None:
            target_col = self.config.model.TARGET_COL
        
        submission = pd.DataFrame({
            id_col: test_df[id_col],
            target_col: predictions
        })
        
        if output_path is None:
            output_path = f"{self.config.paths.OUTPUT_DIR}{self.config.paths.SUBMISSION_FILE}"
        
        submission.to_csv(output_path, index=False)
        logger.info(f"Submission saved to {output_path}")
        
        return submission
    
    def run_inference(
        self,
        test_df: pd.DataFrame,
        preprocessor=None,
        model_dir: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Complete inference pipeline
        
        Args:
            test_df: Test dataframe
            preprocessor: Data preprocessor (optional if already set)
            model_dir: Directory with saved models (optional if models already loaded)
            
        Returns:
            Submission dataframe
        """
        logger.info("=" * 60)
        logger.info("Starting inference pipeline")
        logger.info("=" * 60)
        
        # Load models if directory provided
        if model_dir:
            self.load_models(model_dir)
        
        # Preprocess data if preprocessor provided
        # Preprocessor can be:
        # - an instance of DataPreprocessor (already fitted),
        # - a path to a saved preprocessor artifact, or
        # - None (assume test_df already preprocessed)
        if preprocessor:
            if isinstance(preprocessor, str):
                # load saved artifact
                self.preprocessor = DataPreprocessor.load(preprocessor)
            else:
                self.preprocessor = preprocessor

            # Use inference transform (does not peek at labels)
            X_test = self.preprocessor.transform_for_inference(test_df)
        else:
            # Assume test_df is already preprocessed
            X_test = test_df.drop(columns=[self.config.model.ID_COL])
        
        # Make predictions
        predictions = self.predict_ensemble(X_test)
        
        # Create submission
        submission = self.create_submission(test_df, predictions)
        
        logger.info("Inference pipeline completed")
        
        return submission


def save_models(models: List, model_dir: str, prefix: str = "model"):
    """
    Save trained models to disk
    
    Args:
        models: List of trained models
        model_dir: Directory to save models
        prefix: Prefix for model filenames
    """
    os.makedirs(model_dir, exist_ok=True)
    
    for i, model in enumerate(models):
        path = os.path.join(model_dir, f"{prefix}_fold_{i}.pkl")
        with open(path, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"Saved model to {path}")


def create_inference_pipeline(config):
    """
    Factory function to create inference pipeline
    
    Args:
        config: Configuration object
        
    Returns:
        InferencePipeline instance
    """
    return InferencePipeline(config)



import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import logging
import warnings

# Import custom modules
#from config import Config
#from data_preprocessing import create_preprocessor
import os
#from model_training import create_trainer
#from model_evaluation import create_evaluator
#from inference_pipeline import save_models

warnings.filterwarnings('ignore')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def training_pipeline(
    train_path: str = None,
    test_path: str = None,
    config: Config = None
):
    """
    Complete training pipeline with hyperparameter optimization
    
    Args:
        train_path: Path to training data
        test_path: Path to test data
        config: Configuration object (creates default if None)
        
    Returns:
        Dictionary containing:
            - predictions: Test predictions
            - models: List of trained models
            - study_results: Hyperparameter optimization results
            - feature_importances: Feature importance scores
            - best_model_name: Name of best performing model
    """
    # Initialize configuration
    if config is None:
        config = Config()
    
    if train_path is None:
        train_path = config.paths.TRAIN_DATA_PATH
    if test_path is None:
        test_path = config.paths.TEST_DATA_PATH
    
    logger.info("=" * 80)
    logger.info("STARTING TRAINING PIPELINE")
    logger.info("=" * 80)
    
    # ========================================
    # STEP 1: DATA PREPROCESSING
    # ========================================
    logger.info("\nSTEP 1: Data Preprocessing")
    logger.info("-" * 80)
    
    preprocessor = create_preprocessor(config)

    # Load training data and fit preprocessor (do not use test data here)
    train_df = preprocessor.load_data(train_path)
    X_train, y_train = preprocessor.fit_transform(train_df)
    cat_features_indices = preprocessor.cat_features_indices

    # Save the fitted preprocessor artifact for later inference
    preproc_path = os.path.join(config.paths.MODEL_DIR, 'preprocessor.joblib')
    preprocessor.save(preproc_path)
    
    logger.info(f"Training set shape: {X_train.shape}")
    logger.info(f"Target distribution:\n{y_train.value_counts(normalize=True)}")
    
    # ========================================
    # STEP 2: HYPERPARAMETER OPTIMIZATION
    # ========================================
    logger.info("\nSTEP 2: Hyperparameter Optimization")
    logger.info("-" * 80)
    
    trainer = create_trainer(config, cat_features_indices)
    study_results = trainer.optimize_all_models(X_train, y_train)
    
    # ========================================
    # STEP 3: TRAIN FINAL MODEL
    # ========================================
    logger.info("\nSTEP 3: Training Final Model with K-Fold CV")
    logger.info("-" * 80)
    
    # Train final models (no test data passed)
    predictions, fold_models, feature_importances = trainer.train_final_model(
        X_train, y_train
    )
    
    # Save trained models
    save_models(
        fold_models,
        config.paths.MODEL_DIR,
        prefix=f"model_{trainer.best_model_name}"
    )
    
    # ========================================
    # STEP 4: EVALUATION AND VISUALIZATION
    # ========================================
    logger.info("\nSTEP 4: Evaluation and Visualization")
    logger.info("-" * 80)
    
    evaluator = create_evaluator(config)
    
    # Get validation set for SHAP analysis (use last fold)
    kf = StratifiedKFold(
        n_splits=config.model.N_FOLDS_SUB,
        shuffle=True,
        random_state=config.model.RANDOM_SEED
    )
    
    for fold_idx, (_, val_idx) in enumerate(kf.split(X_train, y_train)):
        if fold_idx == config.model.N_FOLDS_SUB - 1:
            X_val = X_train.iloc[val_idx]
            break
    
    # Run evaluation
    evaluator.evaluate_and_visualize(
        study_results=study_results,
        final_model=fold_models[-1],  # Use last fold model for SHAP
        X_val=X_val,
        feature_names=X_train.columns.tolist(),
        feature_importances=feature_importances,
        model_name=trainer.best_model_name
    )
    
    # NOTE: Creating a submission file is part of the inference step and is
    # handled by `inference_pipeline.py`. The training pipeline saves models
    # and the fitted preprocessor for later use in inference.
    
    # ========================================
    # SUMMARY
    # ========================================
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)
    logger.info(f"Best Model: {trainer.best_model_name.upper()}")
    logger.info(f"Best CV AUC: {study_results[trainer.best_model_name]['best_score']:.4f}")
    logger.info(f"Models saved to: {config.paths.MODEL_DIR}")
    logger.info(f"Visualizations saved to: {config.paths.OUTPUT_DIR}")
    logger.info("=" * 80)
    
    return {
        'predictions': predictions,
        'models': fold_models,
        'study_results': study_results,
        'feature_importances': feature_importances,
        'best_model_name': trainer.best_model_name,
        'feature_names': X_train.columns.tolist(),
        'config': config
    }


import os
import pandas as pd
import logging
import warnings

# Import custom modules
#from config import Config
#from data_preprocessing import DataPreprocessor
#from inference_pipeline import create_inference_pipeline

warnings.filterwarnings('ignore')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def inference_pipeline(
    test_path: str = None,
    model_dir: str = None,
    output_path: str = None,
    config: Config = None,
    preprocessor_path: str = None
):
    """
    Complete inference pipeline for making predictions
    
    Args:
        test_path: Path to test data
        model_dir: Directory containing saved models
        output_path: Path to save submission file
        config: Configuration object (creates default if None)
        
    Returns:
        Submission dataframe with predictions
    """
    # Initialize configuration
    if config is None:
        config = Config()
    
    if test_path is None:
        test_path = config.paths.TEST_DATA_PATH
    if model_dir is None:
        model_dir = config.paths.MODEL_DIR
    if output_path is None:
        output_path = f"{config.paths.OUTPUT_DIR}{config.paths.SUBMISSION_FILE}"
    
    logger.info("=" * 80)
    logger.info("STARTING INFERENCE PIPELINE")
    logger.info("=" * 80)
    
    # ========================================
    # STEP 1: LOAD AND PREPROCESS DATA
    # ========================================
    logger.info("\nSTEP 1: Loading and Preprocessing Data")
    logger.info("-" * 80)
    
    # Load test data
    test_df = pd.read_csv(test_path)
    logger.info(f"Test data loaded: {test_df.shape}")
    
    # Load the fitted preprocessor artifact created during training
    if preprocessor_path:
        preproc_path = preprocessor_path
    else:
        preproc_path = os.path.join(config.paths.MODEL_DIR, 'preprocessor.joblib')

    if not os.path.exists(preproc_path):
        raise FileNotFoundError(
            f"Preprocessor artifact not found at {preproc_path}. "
            "Run the training pipeline to create and save the fitted preprocessor, "
            "or pass a valid `preprocessor_path` to this function."
        )

    preprocessor = DataPreprocessor.load(preproc_path)
    logger.info(f"Loaded preprocessor from {preproc_path}")
    X_test = preprocessor.transform_for_inference(test_df)
    logger.info(f"Test features prepared: {X_test.shape}")
    
    # ========================================
    # STEP 2: LOAD MODELS AND MAKE PREDICTIONS
    # ========================================
    logger.info("STEP 2: Loading Models and Making Predictions")
    
    inference = create_inference_pipeline(config)
    inference.load_models(model_dir)
    
    # Make predictions
    predictions = inference.predict_ensemble(X_test)
    
    logger.info(f"Predictions generated for {len(predictions)} samples")
    logger.info(f"Prediction statistics:")
    logger.info(f"  Mean: {predictions.mean():.4f}")
    logger.info(f"  Std:  {predictions.std():.4f}")
    logger.info(f"  Min:  {predictions.min():.4f}")
    logger.info(f"  Max:  {predictions.max():.4f}")
    
    # ========================================
    # STEP 3: CREATE SUBMISSION
    # ========================================
    logger.info("STEP 3: Creating Submission File")
    
    submission = inference.create_submission(
        test_df=test_df,
        predictions=predictions,
        output_path=output_path
    )
    
    logger.info(f"Submission preview:")
    logger.info(f"\n{submission.head(10)}")
    
    # ========================================
    # SUMMARY
    # ========================================
    logger.info("INFERENCE PIPELINE COMPLETED SUCCESSFULLY")
    logger.info(f"Number of predictions: {len(predictions)}")
    logger.info(f"Submission saved to: {output_path}")
    
    return submission





# ============================================
# SETUP - Add this at the top of your Kaggle notebook
# ============================================

# Import all required modules (assuming they're in the same directory)
import sys
sys.path.append('/kaggle/working/')

#from config import Config
#from train_pipeline import training_pipeline

# ============================================
# CONFIGURATION
# ============================================

# Create configuration object
config = Config()

# Update paths for Kaggle environment
config.paths.TRAIN_DATA_PATH = "/kaggle/input/playground-series-s5e12/train.csv"
config.paths.TEST_DATA_PATH = "/kaggle/input/playground-series-s5e12/test.csv"
config.paths.OUTPUT_DIR = "/kaggle/working/"
config.paths.MODEL_DIR = "/kaggle/working/models/"

# ============================================
# OPTIMIZATION SETTINGS
# ============================================

# GPU Configuration
# Set to True if you selected GPU accelerator in Kaggle
config.model.USE_GPU = False  # Change to True for GPU

# Models to test
config.model.MODELS_TO_TEST = ['xgboost', 'lightgbm', 'catboost']

# Hyperparameter optimization settings
# Faster settings for quick iteration
config.model.N_TRIALS = 15  # Increase for better results (e.g., 30)
config.model.N_FOLDS_OPT = 3  # Folds for optimization
config.model.MAX_OPT_SAMPLES = 100000  # Sample size for Optuna

# Final model training settings
# Higher quality settings for submission
config.model.N_FOLDS_SUB = 5  # Increase for more stable predictions (e.g., 10)
config.model.MAX_ITERATIONS = 1500
config.model.EARLY_STOPPING_ROUNDS = 50

# ============================================
# RUN TRAINING PIPELINE
# ============================================

print("Starting training pipeline...")
print("=" * 80)

results = training_pipeline(config=config)

# ============================================
# RESULTS SUMMARY
# ============================================

print("\n" + "=" * 80)
print("TRAINING COMPLETED")
print("=" * 80)
print(f"\nBest Model: {results['best_model_name'].upper()}")
print(f"CV AUC Score: {results['study_results'][results['best_model_name']]['best_score']:.4f}")

print("\nAll Model Scores:")
for model_name, result in results['study_results'].items():
    print(f"  {model_name.upper()}: {result['best_score']:.4f}")

print("\nTop 10 Features:")
import pandas as pd
fi_df = pd.DataFrame({
    'feature': results['feature_names'],
    'importance': results['feature_importances']
})
fi_df = fi_df.sort_values('importance', ascending=False).head(10)
print(fi_df.to_string(index=False))


print(f"✓ Models saved to: {config.paths.MODEL_DIR}")
print(f"✓ Fitted preprocessor artifact: {config.paths.MODEL_DIR}preprocessor.joblib")
print(f"✓ Visualizations saved to: {config.paths.OUTPUT_DIR}")
print("  - model_comparison.png")
print("  - feature_importance_best_model.png")
print("  - shap_summary_best_model.png")



# Import all required modules
import sys
sys.path.append('/kaggle/working/')

#from config import Config
#from predict_pipeline import inference_pipeline

# ============================================
# CONFIGURATION
# ============================================

# Create configuration object
config = Config()

# Update paths for Kaggle environment
config.paths.TEST_DATA_PATH = "/kaggle/input/playground-series-s5e12/test.csv"

# Path to saved models
# Option 1: If models are in a Kaggle dataset you created
#config.paths.MODEL_DIR = "/kaggle/input/your-model-dataset/models/"

# Option 2: If models are in the working directory from training
config.paths.MODEL_DIR = "/kaggle/working/models/"

# Output path
config.paths.OUTPUT_DIR = "/kaggle/working/"
config.paths.SUBMISSION_FILE = "submission.csv"

# Path to the fitted preprocessor artifact (saved during training)
preproc_path = f"{config.paths.MODEL_DIR}preprocessor.joblib"

import os
if not os.path.exists(preproc_path):
    print("WARNING: Preprocessor artifact not found at:", preproc_path)
    print("Make sure to include the fitted preprocessor in your model dataset or copy it to the model directory.")

# ============================================
# RUN INFERENCE PIPELINE
# ============================================

print("Starting inference pipeline...")
print("=" * 80)

submission = inference_pipeline(config=config)

# ============================================
# RESULTS SUMMARY
# ============================================

print("\n" + "=" * 80)
print("INFERENCE COMPLETED")
print("=" * 80)

print("\nSubmission Statistics:")
print(f"  Number of predictions: {len(submission)}")
print(f"  Mean prediction: {submission[config.model.TARGET_COL].mean():.4f}")
print(f"  Std prediction: {submission[config.model.TARGET_COL].std():.4f}")
print(f"  Min prediction: {submission[config.model.TARGET_COL].min():.4f}")
print(f"  Max prediction: {submission[config.model.TARGET_COL].max():.4f}")

print("\nFirst 10 predictions:")
print(submission.head(10))

print("\nPrediction distribution:")
print(submission[config.model.TARGET_COL].describe())

print("\n" + "=" * 80)
print(f"✓ Submission saved to: {config.paths.OUTPUT_DIR}{config.paths.SUBMISSION_FILE}")
print(f"✓ Preprocessor artifact used (expected): {preproc_path}")
print("=" * 80)

# ============================================
# VALIDATION CHECKS
# ============================================

print("\n" + "=" * 80)
print("VALIDATION CHECKS:")
print("=" * 80)

# Check for any issues
issues = []

# Check for NaN values
if submission[config.model.TARGET_COL].isna().any():
    issues.append("⚠ Warning: NaN values detected in predictions!")

# Check prediction range
if (submission[config.model.TARGET_COL] < 0).any():
    issues.append("⚠ Warning: Negative predictions detected!")

if (submission[config.model.TARGET_COL] > 1).any():
    issues.append("⚠ Warning: Predictions > 1 detected!")

# Check ID column
import pandas as pd
test_df = pd.read_csv(config.paths.TEST_DATA_PATH)
if not submission[config.model.ID_COL].equals(test_df[config.model.ID_COL]):
    issues.append("⚠ Warning: ID column mismatch with test data!")

if len(issues) == 0:
    print("✓ All validation checks passed!")
else:
    for issue in issues:
        print(issue)

print("=" * 80)

# ============================================
# OPTIONAL: Quick Analysis
# ============================================

print("\n" + "=" * 80)
print("QUICK ANALYSIS:")
print("=" * 80)

# Distribution of predictions
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
sns.histplot(submission[config.model.TARGET_COL], bins=50, kde=True)
plt.title('Distribution of Predictions')
plt.xlabel('Predicted Probability')
plt.ylabel('Count')

plt.subplot(1, 2, 2)
plt.boxplot(submission[config.model.TARGET_COL])
plt.title('Prediction Box Plot')
plt.ylabel('Predicted Probability')

plt.tight_layout()
plt.savefig('prediction_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Saved prediction distribution plot")
plt.show()




