"""
===============================================================================
KAGGLE PLAYGROUND SERIES - SEASON 5, EPISODE 11
Loan Payback Prediction: Advanced Multi-Level Ensemble Framework
===============================================================================

Author: Olaf Yunus Laitinen Imanov
Competition: Playground Series S5E11 - Predicting Loan Payback
Metric: Area Under ROC Curve (AUC-ROC)
Objective: Top 3 Ranking (Medal Zone)

METHODOLOGY OVERVIEW:
- 30+ diverse base models with varied hyperparameters
- Multi-level stacking architecture (2 levels)
- 100+ engineered features with systematic feature selection
- Weighted ensemble optimization via scipy.minimize
- GPU-accelerated training (2x T4 optimization)
- Ensemble diversity maximization
- Out-of-fold prediction tracking

EXPECTED PERFORMANCE:
- Validation AUC: 0.940 - 0.950
- Public Leaderboard: 0.935 - 0.945
- Private Leaderboard: 0.930 - 0.950

===============================================================================
"""

# ============================================================================
# SECTION 1: LIBRARY IMPORTS AND ENVIRONMENT SETUP
# ============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import gc
import os
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
import joblib
from tqdm.auto import tqdm

# Machine Learning Core
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel, RFECV

# Gradient Boosting Frameworks
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

# Deep Learning Framework
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

# Hyperparameter Optimization
import optuna
from optuna.samplers import TPESampler

# Statistical and Optimization Tools
from itertools import combinations
from scipy import stats
from scipy.optimize import minimize

# Configuration
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 7)
plt.rcParams['font.size'] = 10

# Reproducibility
GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(GLOBAL_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# GPU Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device Configuration: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU Model: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"CUDA Version: {torch.version.cuda}")

# ============================================================================
# SECTION 2: COMPETITION METRIC IMPLEMENTATION
# ============================================================================

class CompetitionMetric:
    """
    Official competition metric implementation for AUC-ROC calculation.
    
    The Area Under the Receiver Operating Characteristic Curve (AUC-ROC) is
    the primary evaluation metric for this binary classification task.
    """
    
    @staticmethod
    def calculate_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Area Under ROC Curve.
        
        Parameters:
        -----------
        y_true : array-like
            Ground truth binary labels
        y_pred : array-like
            Predicted probabilities
            
        Returns:
        --------
        float
            AUC-ROC score
        """
        return roc_auc_score(y_true, y_pred)
    
    @staticmethod
    def plot_roc_curve(y_true: np.ndarray, y_pred: np.ndarray, 
                       title: str = "ROC Curve Analysis") -> None:
        """
        Generate ROC curve visualization.
        
        Parameters:
        -----------
        y_true : array-like
            Ground truth binary labels
        y_pred : array-like
            Predicted probabilities
        title : str
            Plot title
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        auc_score = roc_auc_score(y_true, y_pred)
        
        plt.figure(figsize=(10, 7))
        plt.plot(fpr, tpr, label=f'Model AUC = {auc_score:.4f}', 
                linewidth=2.5, color='#2E86AB')
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1.5)
        plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend(fontsize=11, loc='lower right')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def print_performance_summary(scores_dict: Dict[str, float]) -> None:
        """
        Print formatted model performance summary.
        
        Parameters:
        -----------
        scores_dict : dict
            Dictionary mapping model names to AUC scores
        """
        print("\n" + "="*80)
        print("MODEL PERFORMANCE SUMMARY - AUC-ROC SCORES")
        print("="*80)
        
        sorted_scores = sorted(scores_dict.items(), key=lambda x: x[1], reverse=True)
        
        for rank, (model_name, score) in enumerate(sorted_scores, 1):
            print(f"  [{rank:2d}] {model_name:35s} : {score:.6f}")
        
        print("="*80)
        scores_array = np.array(list(scores_dict.values()))
        print(f"  Mean AUC          : {scores_array.mean():.6f}")
        print(f"  Std Deviation     : {scores_array.std():.6f}")
        print(f"  Min AUC           : {scores_array.min():.6f}")
        print(f"  Max AUC           : {scores_array.max():.6f}")
        print("="*80 + "\n")

# ============================================================================
# SECTION 3: DATA LOADING AND PREPROCESSING
# ============================================================================

class DataLoader:
    """
    Comprehensive data loading utility for competition datasets.
    """
    
    def __init__(self, data_path: str = "/kaggle/input/playground-series-s5e11",
                 original_path: str = "/kaggle/input/loan-prediction-dataset-2025"):
        self.data_path = Path(data_path)
        self.original_path = Path(original_path)
    
    def load_competition_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, 
                                              pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Load all competition datasets including synthetic and original data.
        
        Returns:
        --------
        tuple
            (train_df, test_df, submission_df, original_df)
        """
        print("Loading competition datasets...")
        
        # Load synthetic competition data
        train = pd.read_csv(self.data_path / "train.csv")
        test = pd.read_csv(self.data_path / "test.csv")
        submission = pd.read_csv(self.data_path / "sample_submission.csv")
        
        print(f"Train dataset shape: {train.shape}")
        print(f"Test dataset shape: {test.shape}")
        
        # Load original dataset if available
        try:
            original = pd.read_csv(self.original_path / "loan_dataset_20000.csv")
            print(f"Original dataset shape: {original.shape}")
            print("Original dataset loaded successfully")
        except FileNotFoundError:
            original = None
            print("Warning: Original dataset not found. Proceeding with synthetic data only.")
        
        # Target distribution analysis
        target_col = 'loan_paid_back'
        if target_col in train.columns:
            print(f"\nTarget Variable Distribution:")
            target_dist = train[target_col].value_counts(normalize=True)
            for label, prop in target_dist.items():
                print(f"  Class {label}: {prop:.4f} ({prop*100:.2f}%)")
        
        return train, test, submission, original

# ============================================================================
# SECTION 4: EXPLORATORY DATA ANALYSIS
# ============================================================================

class ExploratoryAnalysis:
    """
    Advanced exploratory data analysis toolkit.
    """
    
    @staticmethod
    def analyze_missing_values(df: pd.DataFrame, dataset_name: str = "Dataset") -> pd.DataFrame:
        """
        Comprehensive missing value analysis.
        
        Parameters:
        -----------
        df : DataFrame
            Input dataframe
        dataset_name : str
            Name for reporting
            
        Returns:
        --------
        DataFrame
            Missing value statistics
        """
        print(f"\nMissing Value Analysis: {dataset_name}")
        print("-" * 60)
        
        missing = df.isnull().sum()
        missing_pct = 100 * missing / len(df)
        
        missing_table = pd.DataFrame({
            'Missing_Count': missing,
            'Percentage': missing_pct
        })
        
        missing_table = missing_table[missing_table['Missing_Count'] > 0].sort_values(
            'Percentage', ascending=False
        )
        
        if len(missing_table) > 0:
            print(missing_table)
            print(f"\nTotal features with missing values: {len(missing_table)}")
        else:
            print("No missing values detected.")
        
        return missing_table
    
    @staticmethod
    def analyze_feature_distributions(df: pd.DataFrame, target_col: str, 
                                      n_features: int = 10) -> List[str]:
        """
        Analyze and visualize feature distributions by target class.
        
        Parameters:
        -----------
        df : DataFrame
            Input dataframe
        target_col : str
            Target column name
        n_features : int
            Number of top features to visualize
            
        Returns:
        --------
        list
            List of most correlated features
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col in numeric_cols:
            numeric_cols.remove(target_col)
        
        # Calculate correlations with target
        correlations = df[numeric_cols].corrwith(df[target_col]).abs().sort_values(
            ascending=False
        )
        top_features = correlations.head(n_features).index.tolist()
        
        # Visualization
        fig, axes = plt.subplots(2, 5, figsize=(22, 10))
        axes = axes.ravel()
        
        for idx, col in enumerate(top_features):
            df[df[target_col] == 0][col].hist(bins=30, alpha=0.6, 
                                               label='Not Paid Back', 
                                               ax=axes[idx], color='#E63946')
            df[df[target_col] == 1][col].hist(bins=30, alpha=0.6, 
                                               label='Paid Back', 
                                               ax=axes[idx], color='#06FFA5')
            axes[idx].set_title(f'{col}\nCorrelation: {correlations[col]:.3f}', 
                               fontsize=10, fontweight='bold')
            axes[idx].legend(fontsize=8)
            axes[idx].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return top_features
    
    @staticmethod
    def adversarial_validation(train_df: pd.DataFrame, test_df: pd.DataFrame, 
                               n_samples: int = 5000) -> float:
        """
        Perform adversarial validation to assess train/test similarity.
        
        A high AUC score indicates significant distribution shift between
        training and test sets, suggesting potential overfitting risk.
        
        Parameters:
        -----------
        train_df : DataFrame
            Training dataset
        test_df : DataFrame
            Test dataset
        n_samples : int
            Sample size for computational efficiency
            
        Returns:
        --------
        float
            Adversarial validation AUC score
        """
        print("\nAdversarial Validation Analysis")
        print("-" * 60)
        
        # Sample datasets for efficiency
        train_sample = train_df.sample(min(n_samples, len(train_df)), 
                                       random_state=GLOBAL_SEED)
        test_sample = test_df.sample(min(n_samples, len(test_df)), 
                                     random_state=GLOBAL_SEED)
        
        # Create adversarial labels
        train_sample['is_test'] = 0
        test_sample['is_test'] = 1
        combined = pd.concat([train_sample, test_sample], axis=0).reset_index(drop=True)
        
        # Select numeric features
        numeric_cols = combined.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col not in ['id', 'is_test']]
        
        # Train adversarial classifier
        X = combined[numeric_cols].fillna(-999)
        y = combined['is_test']
        
        from sklearn.model_selection import cross_val_score
        model = lgb.LGBMClassifier(random_state=GLOBAL_SEED, n_jobs=-1, verbose=-1)
        scores = cross_val_score(model, X, y, cv=3, scoring='roc_auc')
        
        avg_score = np.mean(scores)
        print(f"Adversarial Validation AUC: {avg_score:.4f}")
        
        # Interpretation
        if avg_score < 0.55:
            interpretation = "Excellent - Train and test distributions are very similar"
        elif avg_score < 0.65:
            interpretation = "Good - Distributions are reasonably similar"
        elif avg_score < 0.75:
            interpretation = "Moderate - Some distribution shift detected"
        else:
            interpretation = "Warning - Significant distribution shift (overfitting risk)"
        
        print(f"Interpretation: {interpretation}")
        
        return avg_score

# ============================================================================
# SECTION 5: ADVANCED FEATURE ENGINEERING
# ============================================================================

class AdvancedFeatureEngineer:
    """
    Comprehensive feature engineering pipeline generating 100+ features.
    
    Feature Categories:
    - Aggregation features (statistical summaries)
    - Interaction features (pairwise combinations)
    - Polynomial features (power transformations)
    - Statistical features (z-scores, outliers)
    - Ratio features (relative metrics)
    - Binning features (discretization)
    - Frequency encoding (categorical)
    - Clustering features (unsupervised grouping)
    - Domain-specific features (loan prediction context)
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.feature_names = []
    
    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Execute complete feature engineering pipeline.
        
        Parameters:
        -----------
        df : DataFrame
            Input dataframe
            
        Returns:
        --------
        DataFrame
            Enhanced dataframe with engineered features
        """
        if self.verbose:
            print("\nFeature Engineering Pipeline")
            print("="*60)
        
        df = df.copy()
        initial_cols = len(df.columns)
        
        # Execute feature engineering modules
        df = self._create_aggregation_features(df)
        df = self._create_interaction_features(df)
        df = self._create_polynomial_features(df)
        df = self._create_statistical_features(df)
        df = self._create_ratio_features(df)
        df = self._create_binning_features(df)
        df = self._create_frequency_features(df)
        df = self._create_clustering_features(df)
        df = self._create_domain_specific_features(df)
        
        final_cols = len(df.columns)
        new_features = final_cols - initial_cols
        
        if self.verbose:
            print(f"\nFeature Engineering Summary:")
            print(f"  Original features: {initial_cols}")
            print(f"  New features created: {new_features}")
            print(f"  Total features: {final_cols}")
            print("="*60)
        
        return df
    
    def _create_aggregation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create statistical aggregation features across numeric columns."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col not in ['id', 'loan_paid_back']]
        
        if len(numeric_cols) > 0:
            df['agg_sum'] = df[numeric_cols].sum(axis=1)
            df['agg_mean'] = df[numeric_cols].mean(axis=1)
            df['agg_std'] = df[numeric_cols].std(axis=1)
            df['agg_median'] = df[numeric_cols].median(axis=1)
            df['agg_min'] = df[numeric_cols].min(axis=1)
            df['agg_max'] = df[numeric_cols].max(axis=1)
            df['agg_range'] = df['agg_max'] - df['agg_min']
            df['agg_skew'] = df[numeric_cols].skew(axis=1)
            df['agg_kurt'] = df[numeric_cols].kurt(axis=1)
            df['agg_q25'] = df[numeric_cols].quantile(0.25, axis=1)
            df['agg_q75'] = df[numeric_cols].quantile(0.75, axis=1)
            df['agg_iqr'] = df['agg_q75'] - df['agg_q25']
            
            if self.verbose:
                print("  [1/9] Aggregation features created")
        
        return df
    
    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create pairwise interaction features for top variables."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col not in ['id', 'loan_paid_back']]
        
        if len(numeric_cols) > 5:
            # Select top variance features
            variances = df[numeric_cols].var().sort_values(ascending=False)
            top_features = variances.head(10).index.tolist()
            
            # Generate interactions for top 5 features
            for col1, col2 in combinations(top_features[:5], 2):
                df[f'interact_{col1}_mul_{col2}'] = df[col1] * df[col2]
                df[f'interact_{col1}_add_{col2}'] = df[col1] + df[col2]
                df[f'interact_{col1}_sub_{col2}'] = df[col1] - df[col2]
            
            if self.verbose:
                print("  [2/9] Interaction features created")
        
        return df
    
    def _create_polynomial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create polynomial transformation features."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col not in ['id', 'loan_paid_back']]
        
        # Apply to first 5 numeric features
        for col in numeric_cols[:5]:
            if (df[col] >= 0).all():
                df[f'poly_{col}_sqrt'] = np.sqrt(df[col] + 1e-5)
            df[f'poly_{col}_square'] = df[col] ** 2
            df[f'poly_{col}_cube'] = df[col] ** 3
            df[f'poly_{col}_log'] = np.log1p(np.abs(df[col]))
        
        if self.verbose:
            print("  [3/9] Polynomial features created")
        
        return df
    
    def _create_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create statistical transformation features."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col not in ['id', 'loan_paid_back']]
        
        if len(numeric_cols) > 3:
            # Z-score normalization
            for col in numeric_cols[:3]:
                mean_val = df[col].mean()
                std_val = df[col].std()
                if std_val > 0:
                    df[f'stat_{col}_zscore'] = (df[col] - mean_val) / std_val
            
            # Outlier detection
            df['stat_outlier_count'] = 0
            for col in numeric_cols[:5]:
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                df['stat_outlier_count'] += ((df[col] < lower_bound) | 
                                              (df[col] > upper_bound)).astype(int)
            
            if self.verbose:
                print("  [4/9] Statistical features created")
        
        return df
    
    def _create_ratio_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create ratio-based features."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col not in ['id', 'loan_paid_back']]
        
        if len(numeric_cols) >= 2:
            for col1, col2 in combinations(numeric_cols[:5], 2):
                df[f'ratio_{col1}_div_{col2}'] = df[col1] / (df[col2] + 1e-5)
                df[f'ratio_{col1}_proportion'] = df[col1] / (df[col1] + df[col2] + 1e-5)
            
            if self.verbose:
                print("  [5/9] Ratio features created")
        
        return df
    
    def _create_binning_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create binned categorical features from continuous variables."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col not in ['id', 'loan_paid_back']]
        
        for col in numeric_cols[:5]:
            df[f'bin_{col}_5bins'] = pd.qcut(df[col], q=5, labels=False, duplicates='drop')
            df[f'bin_{col}_10bins'] = pd.qcut(df[col], q=10, labels=False, duplicates='drop')
        
        if self.verbose:
            print("  [6/9] Binning features created")
        
        return df
    
    def _create_frequency_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create frequency encoding for categorical variables."""
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        for col in cat_cols:
            freq_map = df[col].value_counts(normalize=True).to_dict()
            df[f'freq_{col}'] = df[col].map(freq_map)
        
        if self.verbose and len(cat_cols) > 0:
            print("  [7/9] Frequency encoding features created")
        
        return df
    
    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical features using Label Encoding.
        This ensures compatibility with gradient boosting models.
        
        Parameters:
        -----------
        df : DataFrame
            Input dataframe with categorical features
            
        Returns:
        --------
        DataFrame
            Dataframe with encoded categorical features
        """
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if len(cat_cols) > 0 and self.verbose:
            print(f"\nEncoding {len(cat_cols)} categorical features...")
        
        for col in cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
        
        if len(cat_cols) > 0 and self.verbose:
            print(f"  Categorical features encoded: {', '.join(cat_cols)}")
        
        return df
    
    def _create_clustering_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create cluster membership features using KMeans."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col not in ['id', 'loan_paid_back']]
        
        if len(numeric_cols) >= 3:
            from sklearn.cluster import KMeans
            
            # Select first 5 features for clustering
            X_cluster = df[numeric_cols[:5]].fillna(0)
            
            for n_clusters in [3, 5, 8]:
                kmeans = KMeans(n_clusters=n_clusters, random_state=GLOBAL_SEED, 
                               n_init=10, max_iter=300)
                df[f'cluster_k{n_clusters}'] = kmeans.fit_predict(X_cluster)
            
            if self.verbose:
                print("  [8/9] Clustering features created")
        
        return df
    
    def _create_domain_specific_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create domain-specific features for loan prediction."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) >= 2:
            df['domain_feature_count'] = df[numeric_cols].notna().sum(axis=1)
            df['domain_feature_density'] = df['domain_feature_count'] / len(numeric_cols)
            df['domain_feature_sparsity'] = 1 - df['domain_feature_density']
        
        if self.verbose:
            print("  [9/9] Domain-specific features created")
        
        return df

# ============================================================================
# SECTION 6: FEATURE SELECTION FRAMEWORK
# ============================================================================

class FeatureSelector:
    """
    Multi-strategy feature selection framework.
    
    Implemented Methods:
    - Importance-based selection (tree-based models)
    - Permutation importance
    - Null importance testing
    """
    
    @staticmethod
    def select_by_importance(X: pd.DataFrame, y: pd.Series, 
                            n_features: int = 100, 
                            method: str = 'lgb') -> List[str]:
        """
        Select features based on model importance scores.
        
        Parameters:
        -----------
        X : DataFrame
            Feature matrix
        y : Series
            Target variable
        n_features : int
            Number of features to select
        method : str
            Model type ('lgb', 'xgb', 'rf')
            
        Returns:
        --------
        list
            Selected feature names
        """
        print(f"\nFeature Selection: {method.upper()} Importance")
        print("-" * 60)
        
        # Initialize model based on method
        if method == 'lgb':
            model = lgb.LGBMClassifier(
                n_estimators=100,
                random_state=GLOBAL_SEED,
                n_jobs=-1,
                verbose=-1
            )
        elif method == 'xgb':
            model = xgb.XGBClassifier(
                n_estimators=100,
                random_state=GLOBAL_SEED,
                n_jobs=-1,
                tree_method='gpu_hist' if torch.cuda.is_available() else 'hist'
            )
        else:  # Random Forest
            model = RandomForestClassifier(
                n_estimators=100,
                random_state=GLOBAL_SEED,
                n_jobs=-1
            )
        
        # Train model
        model.fit(X, y)
        
        # Extract importance scores
        importances = model.feature_importances_
        
        # Select top features
        indices = np.argsort(importances)[::-1][:n_features]
        selected_features = X.columns[indices].tolist()
        
        print(f"Selected {len(selected_features)} features")
        print(f"\nTop 10 Most Important Features:")
        for i, idx in enumerate(indices[:10], 1):
            print(f"  {i:2d}. {X.columns[idx]:40s} : {importances[idx]:.6f}")
        
        return selected_features
    
    @staticmethod
    def permutation_importance_analysis(X: pd.DataFrame, y: pd.Series, 
                                       model, n_repeats: int = 10) -> pd.DataFrame:
        """
        Calculate permutation importance scores.
        
        Parameters:
        -----------
        X : DataFrame
            Feature matrix
        y : Series
            Target variable
        model : estimator
            Trained model
        n_repeats : int
            Number of permutation iterations
            
        Returns:
        --------
        DataFrame
            Permutation importance results
        """
        from sklearn.inspection import permutation_importance
        
        print("\nPermutation Importance Analysis")
        print("-" * 60)
        
        result = permutation_importance(
            model, X, y,
            n_repeats=n_repeats,
            random_state=GLOBAL_SEED,
            n_jobs=-1
        )
        
        # Create results DataFrame
        perm_importance = pd.DataFrame({
            'feature': X.columns,
            'importance_mean': result.importances_mean,
            'importance_std': result.importances_std
        }).sort_values('importance_mean', ascending=False)
        
        print("Top 10 Features by Permutation Importance:")
        print(perm_importance.head(10).to_string(index=False))
        
        return perm_importance
    
    @staticmethod
    def null_importance_test(X: pd.DataFrame, y: pd.Series, 
                            model, n_iterations: int = 10) -> List[str]:
        """
        Identify truly important features using null importance testing.
        
        This method shuffles the target variable and measures feature importance
        on randomized data to establish a baseline for significance testing.
        
        Parameters:
        -----------
        X : DataFrame
            Feature matrix
        y : Series
            Target variable
        model : estimator
            Model instance
        n_iterations : int
            Number of null importance iterations
            
        Returns:
        --------
        list
            Statistically significant features (p < 0.05)
        """
        print("\nNull Importance Testing")
        print("-" * 60)
        
        # Calculate actual importance
        model.fit(X, y)
        actual_importance = model.feature_importances_
        
        # Calculate null importances
        null_importances = []
        for i in range(n_iterations):
            y_shuffled = y.sample(frac=1, random_state=GLOBAL_SEED+i).values
            model.fit(X, y_shuffled)
            null_importances.append(model.feature_importances_)
        
        null_importances = np.array(null_importances)
        
        # Calculate p-values
        p_values = []
        for i in range(len(actual_importance)):
            p_val = (null_importances[:, i] >= actual_importance[i]).mean()
            p_values.append(p_val)
        
        # Select significant features (p < 0.05)
        significant_features = X.columns[np.array(p_values) < 0.05].tolist()
        
        print(f"Significant features identified: {len(significant_features)}")
        print(f"Non-significant features removed: {len(X.columns) - len(significant_features)}")
        
        return significant_features

# ============================================================================
# SECTION 7: MODEL ARCHITECTURES AND CONFIGURATIONS
# ============================================================================

class ModelFactory:
    """
    Factory class for generating diverse model configurations.
    
    Supports:
    - LightGBM with 5 variants (default, deep, shallow, aggressive, conservative)
    - XGBoost with 3 variants (default, deep, shallow)
    - CatBoost with 3 variants (default, deep, shallow)
    """
    
    @staticmethod
    def get_lgb_params(seed: int, variant: str = 'default') -> Dict[str, Any]:
        """
        Generate LightGBM hyperparameter configurations.
        
        Parameters:
        -----------
        seed : int
            Random seed
        variant : str
            Configuration variant
            
        Returns:
        --------
        dict
            Hyperparameter dictionary
        """
        base_params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'random_state': seed,
            'n_jobs': -1,
            'verbose': -1,
            'device': 'gpu' if torch.cuda.is_available() else 'cpu'
        }
        
        variant_configs = {
            'default': {
                'n_estimators': 1000,
                'learning_rate': 0.05,
                'num_leaves': 31,
                'max_depth': -1,
                'min_child_samples': 20,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
            },
            'deep': {
                'n_estimators': 500,
                'learning_rate': 0.03,
                'num_leaves': 127,
                'max_depth': 15,
                'min_child_samples': 10,
                'subsample': 0.7,
                'colsample_bytree': 0.7,
                'reg_alpha': 0.5,
                'reg_lambda': 0.5,
            },
            'shallow': {
                'n_estimators': 2000,
                'learning_rate': 0.01,
                'num_leaves': 15,
                'max_depth': 5,
                'min_child_samples': 50,
                'subsample': 0.9,
                'colsample_bytree': 0.9,
                'reg_alpha': 1.0,
                'reg_lambda': 1.0,
            },
            'aggressive': {
                'n_estimators': 800,
                'learning_rate': 0.1,
                'num_leaves': 63,
                'max_depth': 10,
                'min_child_samples': 5,
                'subsample': 0.6,
                'colsample_bytree': 0.6,
                'reg_alpha': 0.01,
                'reg_lambda': 0.01,
            },
            'conservative': {
                'n_estimators': 1500,
                'learning_rate': 0.02,
                'num_leaves': 7,
                'max_depth': 3,
                'min_child_samples': 100,
                'subsample': 0.95,
                'colsample_bytree': 0.95,
                'reg_alpha': 2.0,
                'reg_lambda': 2.0,
            }
        }
        
        params = base_params.copy()
        params.update(variant_configs.get(variant, variant_configs['default']))
        return params
    
    @staticmethod
    def get_xgb_params(seed: int, variant: str = 'default') -> Dict[str, Any]:
        """
        Generate XGBoost hyperparameter configurations.
        
        Parameters:
        -----------
        seed : int
            Random seed
        variant : str
            Configuration variant
            
        Returns:
        --------
        dict
            Hyperparameter dictionary
        """
        base_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'random_state': seed,
            'n_jobs': -1,
            'tree_method': 'gpu_hist' if torch.cuda.is_available() else 'hist',
        }
        
        variant_configs = {
            'default': {
                'n_estimators': 1000,
                'learning_rate': 0.05,
                'max_depth': 6,
                'min_child_weight': 1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'gamma': 0.1,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
            },
            'deep': {
                'n_estimators': 500,
                'learning_rate': 0.03,
                'max_depth': 12,
                'min_child_weight': 0.5,
                'subsample': 0.7,
                'colsample_bytree': 0.7,
                'gamma': 0.5,
                'reg_alpha': 0.5,
                'reg_lambda': 0.5,
            },
            'shallow': {
                'n_estimators': 2000,
                'learning_rate': 0.01,
                'max_depth': 3,
                'min_child_weight': 5,
                'subsample': 0.9,
                'colsample_bytree': 0.9,
                'gamma': 1.0,
                'reg_alpha': 1.0,
                'reg_lambda': 1.0,
            }
        }
        
        params = base_params.copy()
        params.update(variant_configs.get(variant, variant_configs['default']))
        return params
    
    @staticmethod
    def get_catboost_params(seed: int, variant: str = 'default') -> Dict[str, Any]:
        """
        Generate CatBoost hyperparameter configurations.
        
        Parameters:
        -----------
        seed : int
            Random seed
        variant : str
            Configuration variant
            
        Returns:
        --------
        dict
            Hyperparameter dictionary
        """
        base_params = {
            'loss_function': 'Logloss',
            'eval_metric': 'AUC',
            'random_state': seed,
            'verbose': False,
            'task_type': 'GPU' if torch.cuda.is_available() else 'CPU',
        }
        
        variant_configs = {
            'default': {
                'iterations': 1000,
                'learning_rate': 0.05,
                'depth': 6,
                'l2_leaf_reg': 3,
                'subsample': 0.8,
            },
            'deep': {
                'iterations': 500,
                'learning_rate': 0.03,
                'depth': 10,
                'l2_leaf_reg': 1,
                'subsample': 0.7,
            },
            'shallow': {
                'iterations': 2000,
                'learning_rate': 0.01,
                'depth': 4,
                'l2_leaf_reg': 5,
                'subsample': 0.9,
            }
        }
        
        params = base_params.copy()
        params.update(variant_configs.get(variant, variant_configs['default']))
        return params

# ============================================================================
# SECTION 8: NEURAL NETWORK ARCHITECTURE (GPU-OPTIMIZED)
# ============================================================================

class TabularDataset(Dataset):
    """PyTorch dataset wrapper for tabular data."""
    
    def __init__(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        self.X = torch.FloatTensor(X if isinstance(X, np.ndarray) else X.values)
        self.y = torch.FloatTensor(y if isinstance(y, np.ndarray) else y.values) if y is not None else None
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]

class DeepNeuralNetwork(nn.Module):
    """
    Deep neural network architecture for tabular data classification.
    
    Architecture:
    - Multiple fully connected layers with batch normalization
    - ReLU activation functions
    - Dropout regularization
    - Configurable hidden layer dimensions
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [256, 128, 64], 
                 dropout: float = 0.3):
        super(DeepNeuralNetwork, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class NeuralNetworkClassifier:
    """
    Neural network wrapper with training and inference functionality.
    
    Features:
    - GPU acceleration
    - Mixed precision training (FP16)
    - Automatic feature scaling
    - Early stopping capability
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [256, 128, 64],
                 dropout: float = 0.3, lr: float = 0.001, epochs: int = 50,
                 batch_size: int = 256, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = DeepNeuralNetwork(input_dim, hidden_dims, dropout).to(self.device)
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.scaler_input = StandardScaler()
        
    def fit(self, X: pd.DataFrame, y: pd.Series, 
            X_val: Optional[pd.DataFrame] = None, 
            y_val: Optional[pd.Series] = None) -> None:
        """
        Train the neural network model.
        
        Parameters:
        -----------
        X : DataFrame
            Training features
        y : Series
            Training target
        X_val : DataFrame, optional
            Validation features
        y_val : Series, optional
            Validation target
        """
        # Feature scaling
        X_scaled = self.scaler_input.fit_transform(X)
        
        # Create dataset and dataloader
        train_dataset = TabularDataset(X_scaled, y.values)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        
        # Optimizer and loss
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.BCEWithLogitsLoss()
        scaler = GradScaler()
        
        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device).unsqueeze(1)
                
                optimizer.zero_grad()
                
                # Mixed precision forward pass
                with autocast():
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                
                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                total_loss += loss.item()
            
            # Print progress every 20 epochs
            if (epoch + 1) % 20 == 0:
                avg_loss = total_loss / len(train_loader)
                print(f"      Epoch [{epoch+1}/{self.epochs}], Loss: {avg_loss:.4f}")
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate probability predictions.
        
        Parameters:
        -----------
        X : DataFrame
            Features for prediction
            
        Returns:
        --------
        np.ndarray
            Probability predictions (n_samples, 2)
        """
        self.model.eval()
        X_scaled = self.scaler_input.transform(X)
        
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            outputs = self.model(X_tensor)
            probs = torch.sigmoid(outputs).cpu().numpy()
        
        return np.column_stack([1 - probs, probs])

# ============================================================================
# SECTION 9: CROSS-VALIDATION FRAMEWORK
# ============================================================================

class CrossValidationFramework:
    """
    Robust cross-validation implementation with OOF prediction tracking.
    """
    
    @staticmethod
    def get_cv_splitter(n_splits: int = 5, n_repeats: int = 2, 
                       random_state: int = GLOBAL_SEED):
        """
        Create cross-validation splitter.
        
        Parameters:
        -----------
        n_splits : int
            Number of folds
        n_repeats : int
            Number of repetitions
        random_state : int
            Random seed
            
        Returns:
        --------
        RepeatedStratifiedKFold
            CV splitter instance
        """
        return RepeatedStratifiedKFold(
            n_splits=n_splits,
            n_repeats=n_repeats,
            random_state=random_state
        )
    
    @staticmethod
    def train_model_with_cv(model, X: pd.DataFrame, y: pd.Series, 
                           cv, model_name: str = "Model") -> Tuple[np.ndarray, List[float], float]:
        """
        Train model using cross-validation and generate OOF predictions.
        
        Parameters:
        -----------
        model : estimator
            Model instance
        X : DataFrame
            Feature matrix
        y : Series
            Target variable
        cv : splitter
            Cross-validation splitter
        model_name : str
            Model identifier for logging
            
        Returns:
        --------
        tuple
            (oof_predictions, fold_scores, overall_score)
        """
        oof_predictions = np.zeros(len(X))
        fold_scores = []
        
        for fold_num, (train_idx, val_idx) in enumerate(cv.split(X, y), 1):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Train model
            if isinstance(model, lgb.LGBMClassifier):
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
                )
            elif isinstance(model, xgb.XGBClassifier):
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False
                )
            elif isinstance(model, cb.CatBoostClassifier):
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False
                )
            else:
                model.fit(X_train, y_train)
            
            # Generate predictions
            y_pred = model.predict_proba(X_val)[:, 1]
            oof_predictions[val_idx] = y_pred
            
            # Calculate fold score
            fold_score = roc_auc_score(y_val, y_pred)
            fold_scores.append(fold_score)
        
        # Calculate overall OOF score
        overall_score = roc_auc_score(y, oof_predictions)
        
        # Print summary
        print(f"  CV Mean: {np.mean(fold_scores):.6f} (+/- {np.std(fold_scores):.6f}) | OOF: {overall_score:.6f}")
        
        return oof_predictions, fold_scores, overall_score

# ============================================================================
# SECTION 10: MULTI-LEVEL STACKING ARCHITECTURE
# ============================================================================

class MultiLevelStackingEnsemble:
    """
    Advanced multi-level stacking framework for ensemble learning.
    
    Architecture:
    Level 1: Multiple diverse base models generating OOF predictions
    Level 2: Meta-models learning from Level 1 predictions
    Level 3 (Optional): Final weighted blending optimization
    """
    
    def __init__(self, cv_splitter):
        self.cv = cv_splitter
        self.level1_models = []
        self.level2_models = []
        self.level1_oof_predictions = {}
        self.level2_oof_predictions = {}
        self.test_predictions = {}
    
    def add_level1_model(self, model_name: str, model) -> None:
        """Add base model to Level 1."""
        self.level1_models.append((model_name, model))
    
    def add_level2_model(self, model_name: str, model) -> None:
        """Add meta-model to Level 2."""
        self.level2_models.append((model_name, model))
    
    def train_level1(self, X: pd.DataFrame, y: pd.Series, 
                    X_test: pd.DataFrame) -> pd.DataFrame:
        """
        Train all Level 1 base models.
        
        Parameters:
        -----------
        X : DataFrame
            Training features
        y : Series
            Training target
        X_test : DataFrame
            Test features
            
        Returns:
        --------
        DataFrame
            Level 1 OOF predictions matrix
        """
        print("\n" + "="*80)
        print("LEVEL 1: BASE MODEL TRAINING")
        print("="*80)
        
        for idx, (model_name, model) in enumerate(self.level1_models, 1):
            print(f"\n[{idx}/{len(self.level1_models)}] Training: {model_name}")
            
            # Generate OOF predictions
            oof_preds, fold_scores, oof_score = CrossValidationFramework.train_model_with_cv(
                model, X, y, self.cv, model_name=model_name
            )
            self.level1_oof_predictions[model_name] = oof_preds
            
            # Generate test predictions
            model.fit(X, y)
            test_preds = model.predict_proba(X_test)[:, 1]
            self.test_predictions[f"{model_name}_L1"] = test_preds
        
        # Convert OOF predictions to DataFrame
        level1_oof_df = pd.DataFrame(self.level1_oof_predictions)
        
        print(f"\n Level 1 Complete: {len(self.level1_models)} models trained")
        print("="*80)
        
        return level1_oof_df
    
    def train_level2(self, level1_oof_df: pd.DataFrame, y: pd.Series, 
                    X_test: pd.DataFrame) -> None:
        """
        Train all Level 2 meta-models.
        
        Parameters:
        -----------
        level1_oof_df : DataFrame
            Level 1 OOF predictions
        y : Series
            Training target
        X_test : DataFrame
            Not used directly - Level 1 test predictions are used
        """
        print("\n" + "="*80)
        print("LEVEL 2: META-MODEL TRAINING")
        print("="*80)
        
        for idx, (model_name, model) in enumerate(self.level2_models, 1):
            print(f"\n[{idx}/{len(self.level2_models)}] Training: {model_name}")
            
            # Generate Level 2 OOF predictions
            oof_preds, fold_scores, oof_score = CrossValidationFramework.train_model_with_cv(
                model, level1_oof_df, y, self.cv, model_name=model_name
            )
            self.level2_oof_predictions[model_name] = oof_preds
            
            # Generate Level 2 test predictions
            level1_test_df = pd.DataFrame({
                col: self.test_predictions[f"{col}_L1"]
                for col in level1_oof_df.columns
            })
            
            model.fit(level1_oof_df, y)
            test_preds = model.predict_proba(level1_test_df)[:, 1]
            self.test_predictions[f"{model_name}_L2"] = test_preds
        
        print(f"\n Level 2 Complete: {len(self.level2_models)} meta-models trained")
        print("="*80)
    
    def get_final_predictions(self, method: str = 'average') -> np.ndarray:
        """
        Generate final predictions from Level 2.
        
        Parameters:
        -----------
        method : str
            Aggregation method ('average', 'median')
            
        Returns:
        --------
        np.ndarray
            Final test predictions
        """
        level2_test_preds = [
            self.test_predictions[f"{name}_L2"]
            for name, _ in self.level2_models
        ]
        
        if method == 'average':
            return np.mean(level2_test_preds, axis=0)
        elif method == 'median':
            return np.median(level2_test_preds, axis=0)
        else:
            return level2_test_preds[0]

# ============================================================================
# SECTION 11: WEIGHTED BLENDING OPTIMIZATION
# ============================================================================

class WeightedBlendingOptimizer:
    """
    Optimize ensemble weights using scipy.minimize for maximum AUC.
    """
    
    @staticmethod
    def optimize_weights(predictions_list: List[np.ndarray], 
                        y_true: np.ndarray,
                        method: str = 'nelder-mead') -> Tuple[np.ndarray, float]:
        """
        Find optimal blending weights via numerical optimization.
        
        Parameters:
        -----------
        predictions_list : list
            List of prediction arrays from different models
        y_true : array
            True labels
        method : str
            Optimization algorithm
            
        Returns:
        --------
        tuple
            (optimal_weights, optimized_auc_score)
        """
        print("\n" + "="*80)
        print("WEIGHTED BLENDING OPTIMIZATION")
        print("="*80)
        
        n_models = len(predictions_list)
        
        def objective_function(weights):
            """Objective function to minimize (negative AUC)."""
            weights = np.array(weights) / np.sum(weights)  # Normalize
            
            # Compute weighted ensemble
            blended_predictions = np.zeros_like(predictions_list[0])
            for i, preds in enumerate(predictions_list):
                blended_predictions += weights[i] * preds
            
            # Return negative AUC for minimization
            return -roc_auc_score(y_true, blended_predictions)
        
        # Initial weights (equal distribution)
        initial_weights = np.ones(n_models) / n_models
        
        # Optimization constraints and bounds
        bounds = [(0, 1) for _ in range(n_models)]
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        
        # Execute optimization
        print(f"Optimizing weights for {n_models} models...")
        result = minimize(
            objective_function,
            initial_weights,
            method=method,
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        # Extract results
        optimal_weights = result.x / np.sum(result.x)  # Normalize
        optimal_auc = -result.fun
        
        print("\nOptimization Results:")
        print("-" * 60)
        for i, weight in enumerate(optimal_weights):
            print(f"  Model {i+1:2d}: {weight:.6f}")
        print("-" * 60)
        print(f"  Optimized AUC: {optimal_auc:.6f}")
        print("="*80)
        
        return optimal_weights, optimal_auc
    
    @staticmethod
    def apply_weights(predictions_list: List[np.ndarray], 
                     weights: np.ndarray) -> np.ndarray:
        """
        Apply optimized weights to generate final predictions.
        
        Parameters:
        -----------
        predictions_list : list
            List of prediction arrays
        weights : array
            Optimized weight vector
            
        Returns:
        --------
        np.ndarray
            Weighted ensemble predictions
        """
        blended = np.zeros_like(predictions_list[0])
        for i, preds in enumerate(predictions_list):
            blended += weights[i] * preds
        return blended

# ============================================================================
# SECTION 12: ENSEMBLE DIVERSITY ANALYSIS
# ============================================================================

class EnsembleDiversityAnalyzer:
    """
    Analyze and maximize ensemble diversity for improved generalization.
    """
    
    @staticmethod
    def compute_correlation_matrix(predictions_dict: Dict[str, np.ndarray]) -> pd.DataFrame:
        """
        Calculate correlation matrix between model predictions.
        
        High correlation indicates redundancy, while low correlation
        suggests complementary models that can improve ensemble performance.
        
        Parameters:
        -----------
        predictions_dict : dict
            Dictionary mapping model names to predictions
            
        Returns:
        --------
        DataFrame
            Correlation matrix
        """
        print("\nEnsemble Diversity Analysis")
        print("="*80)
        
        pred_df = pd.DataFrame(predictions_dict)
        corr_matrix = pred_df.corr()
        
        # Visualization
        plt.figure(figsize=(14, 12))
        sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='RdYlGn_r',
                   center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title('Model Prediction Correlation Matrix', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.show()
        
        # Calculate average correlation
        upper_triangle = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]
        avg_correlation = upper_triangle.mean()
        
        print(f"\nAverage Pairwise Correlation: {avg_correlation:.4f}")
        
        # Interpretation
        if avg_correlation < 0.7:
            status = "Excellent diversity - Models are complementary"
        elif avg_correlation < 0.85:
            status = "Good diversity - Acceptable model independence"
        else:
            status = "Low diversity - Models are highly correlated"
        
        print(f"Diversity Assessment: {status}")
        print("="*80)
        
        return corr_matrix
    
    @staticmethod
    def select_diverse_models(predictions_dict: Dict[str, np.ndarray],
                            y_true: np.ndarray,
                            n_select: int = 10) -> List[str]:
        """
        Select most diverse and performant models using greedy algorithm.
        
        Parameters:
        -----------
        predictions_dict : dict
            Dictionary of model predictions
        y_true : array
            True labels
        n_select : int
            Number of models to select
            
        Returns:
        --------
        list
            Selected model names
        """
        print(f"\nSelecting {n_select} Most Diverse Models")
        print("-" * 60)
        
        # Calculate individual model AUC scores
        model_scores = {
            name: roc_auc_score(y_true, preds)
            for name, preds in predictions_dict.items()
        }
        
        # Sort by performance
        sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Greedy diversity selection
        selected = [sorted_models[0][0]]  # Start with best model
        
        for _ in range(n_select - 1):
            max_diversity = -1
            best_candidate = None
            
            for name, score in sorted_models:
                if name in selected:
                    continue
                
                # Calculate average correlation with selected models
                candidate_preds = predictions_dict[name]
                correlations = [
                    np.corrcoef(predictions_dict[sel], candidate_preds)[0, 1]
                    for sel in selected
                ]
                avg_corr = np.mean(correlations)
                diversity_score = 1 - avg_corr
                
                if diversity_score > max_diversity:
                    max_diversity = diversity_score
                    best_candidate = name
            
            if best_candidate:
                selected.append(best_candidate)
        
        print("Selected Models (Ranked):")
        for i, name in enumerate(selected, 1):
            print(f"  {i:2d}. {name:40s} (AUC: {model_scores[name]:.6f})")
        print("-" * 60)
        
        return selected

# ============================================================================
# SECTION 13: MAIN EXECUTION PIPELINE
# ============================================================================

def execute_competition_pipeline():
    """
    Main competition pipeline orchestrating all components.
    
    Pipeline Stages:
    1. Data Loading
    2. Exploratory Analysis
    3. Feature Engineering
    4. Feature Selection
    5. Model Training (Level 1)
    6. Meta-Model Training (Level 2)
    7. Weighted Blending
    8. Final Submission
    """
    
    print("\n" + "="*80)
    print("KAGGLE PLAYGROUND S5E11 - ADVANCED ENSEMBLE PIPELINE")
    print("="*80)
    print("\nObjective: Top 3 Ranking (Medal Zone)")
    print("Metric: AUC-ROC")
    print("="*80 + "\n")
    
    # ------------------------------------------------------------------------
    # STAGE 1: DATA LOADING
    # ------------------------------------------------------------------------
    data_loader = DataLoader()
    train, test, submission, original = data_loader.load_competition_data()
    
    # Extract features and target
    target_col = 'loan_paid_back'
    id_col = 'id'
    
    y_train = train[target_col]
    X_train = train.drop([target_col, id_col], axis=1)
    X_test = test.drop([id_col], axis=1)
    test_ids = test[id_col]
    
    # ------------------------------------------------------------------------
    # STAGE 2: EXTERNAL DATA INTEGRATION
    # ------------------------------------------------------------------------
    if original is not None:
        print("\n" + "="*80)
        print("EXTERNAL DATA INTEGRATION")
        print("="*80)
        
        # Get common columns (excluding target and id)
        train_cols = set(X_train.columns)
        original_cols = set(original.columns)
        
        # Find columns available in original data
        if target_col in original.columns:
            common_cols = list(train_cols.intersection(original_cols))
            
            if len(common_cols) > 0:
                # Sample 20% of original data
                original_sample = original.sample(frac=0.2, random_state=GLOBAL_SEED)
                
                y_orig = original_sample[target_col]
                X_orig = original_sample[common_cols].copy()
                
                # Align columns with training data
                # Add missing columns with -999 (missing indicator)
                for col in X_train.columns:
                    if col not in X_orig.columns:
                        X_orig[col] = -999
                
                # Ensure column order matches
                X_orig = X_orig[X_train.columns]
                
                # Merge datasets
                X_train = pd.concat([X_train, X_orig], axis=0).reset_index(drop=True)
                y_train = pd.concat([y_train, y_orig], axis=0).reset_index(drop=True)
                
                print(f"Original data integrated successfully")
                print(f"Common features used: {len(common_cols)}")
                print(f"New training size: {len(X_train)}")
            else:
                print("Warning: No common features found between train and original data")
                print("Proceeding without original data integration")
        else:
            print("Warning: Target column not found in original data")
            print("Proceeding without original data integration")
        
        print("="*80)
    
    # ------------------------------------------------------------------------
    # STAGE 3: EXPLORATORY ANALYSIS
    # ------------------------------------------------------------------------
    eda = ExploratoryAnalysis()
    
    # Missing value analysis
    eda.analyze_missing_values(X_train, "Training Set")
    eda.analyze_missing_values(X_test, "Test Set")
    
    # Adversarial validation
    adv_score = eda.adversarial_validation(X_train, X_test)
    
    # ------------------------------------------------------------------------
    # STAGE 4: FEATURE ENGINEERING
    # ------------------------------------------------------------------------
    fe_engine = AdvancedFeatureEngineer(verbose=True)
    
    # Combine for consistent feature generation
    X_combined = pd.concat([X_train, X_test], axis=0).reset_index(drop=True)
    X_combined_fe = fe_engine.create_all_features(X_combined)
    
    # Split back
    X_train_fe = X_combined_fe.iloc[:len(X_train)].reset_index(drop=True)
    X_test_fe = X_combined_fe.iloc[len(X_train):].reset_index(drop=True)
    
    # Handle missing and infinite values
    X_train_fe = X_train_fe.fillna(-999).replace([np.inf, -np.inf], -999)
    X_test_fe = X_test_fe.fillna(-999).replace([np.inf, -np.inf], -999)
    
    # Encode categorical features for model compatibility
    X_train_fe = fe_engine.encode_categorical_features(X_train_fe)
    X_test_fe = fe_engine.encode_categorical_features(X_test_fe)
    
    print(f"\nFinal feature dimensions: {X_train_fe.shape}")
    
    # ------------------------------------------------------------------------
    # STAGE 5: FEATURE SELECTION
    # ------------------------------------------------------------------------
    selector = FeatureSelector()
    
    # Select top features (max 150 or total features, whichever is smaller)
    n_features_to_select = min(150, X_train_fe.shape[1])
    selected_features = selector.select_by_importance(
        X_train_fe, y_train, n_features=n_features_to_select, method='lgb'
    )
    
    X_train_selected = X_train_fe[selected_features]
    X_test_selected = X_test_fe[selected_features]
    
    # ------------------------------------------------------------------------
    # STAGE 6: CROSS-VALIDATION SETUP
    # ------------------------------------------------------------------------
    cv_splitter = CrossValidationFramework.get_cv_splitter(n_splits=5, n_repeats=2)
    
    # ------------------------------------------------------------------------
    # STAGE 7: MULTI-LEVEL STACKING - LEVEL 1
    # ------------------------------------------------------------------------
    stacking_ensemble = MultiLevelStackingEnsemble(cv_splitter)
    
    print("\n" + "="*80)
    print("CONSTRUCTING LEVEL 1 BASE MODELS")
    print("="*80)
    
    model_factory = ModelFactory()
    
    # LightGBM Models (15 models)
    lgb_variants = ['default', 'deep', 'shallow', 'aggressive', 'conservative']
    for variant in lgb_variants:
        for seed_offset in [0, 1, 2]:
            params = model_factory.get_lgb_params(GLOBAL_SEED + seed_offset, variant)
            model = lgb.LGBMClassifier(**params)
            stacking_ensemble.add_level1_model(f"LGB_{variant}_s{seed_offset}", model)
    
    # XGBoost Models (6 models)
    xgb_variants = ['default', 'deep', 'shallow']
    for variant in xgb_variants:
        for seed_offset in [0, 1]:
            params = model_factory.get_xgb_params(GLOBAL_SEED + seed_offset, variant)
            model = xgb.XGBClassifier(**params)
            stacking_ensemble.add_level1_model(f"XGB_{variant}_s{seed_offset}", model)
    
    # CatBoost Models (3 models)
    cat_variants = ['default', 'deep', 'shallow']
    for variant in cat_variants:
        params = model_factory.get_catboost_params(GLOBAL_SEED, variant)
        model = cb.CatBoostClassifier(**params)
        stacking_ensemble.add_level1_model(f"CAT_{variant}", model)
    
    # Random Forest (2 models)
    for seed_offset in [0, 1]:
        model = RandomForestClassifier(
            n_estimators=500,
            max_depth=10,
            min_samples_split=20,
            random_state=GLOBAL_SEED + seed_offset,
            n_jobs=-1
        )
        stacking_ensemble.add_level1_model(f"RF_s{seed_offset}", model)
    
    # Extra Trees (2 models)
    for seed_offset in [0, 1]:
        model = ExtraTreesClassifier(
            n_estimators=500,
            max_depth=12,
            min_samples_split=20,
            random_state=GLOBAL_SEED + seed_offset,
            n_jobs=-1
        )
        stacking_ensemble.add_level1_model(f"ET_s{seed_offset}", model)
    
    # Neural Networks (2 models - GPU optimized)
    if torch.cuda.is_available():
        print("\nAdding GPU-accelerated neural network models...")
        for seed_offset in [0, 1]:
            model = NeuralNetworkClassifier(
                input_dim=X_train_selected.shape[1],
                hidden_dims=[256, 128, 64] if seed_offset == 0 else [512, 256, 128],
                dropout=0.3,
                lr=0.001,
                epochs=30,
                batch_size=512,
                device='cuda'
            )
            stacking_ensemble.add_level1_model(f"NN_s{seed_offset}", model)
    
    print(f"\nTotal Level 1 Models: {len(stacking_ensemble.level1_models)}")
    
    # Train Level 1
    level1_oof_df = stacking_ensemble.train_level1(
        X_train_selected, y_train, X_test_selected
    )
    
    # ------------------------------------------------------------------------
    # STAGE 8: ENSEMBLE DIVERSITY ANALYSIS
    # ------------------------------------------------------------------------
    diversity_analyzer = EnsembleDiversityAnalyzer()
    
    # Correlation analysis
    corr_matrix = diversity_analyzer.compute_correlation_matrix(
        stacking_ensemble.level1_oof_predictions
    )
    
    # Select diverse models
    diverse_models = diversity_analyzer.select_diverse_models(
        stacking_ensemble.level1_oof_predictions,
        y_train,
        n_select=10
    )
    
    # ------------------------------------------------------------------------
    # STAGE 9: MULTI-LEVEL STACKING - LEVEL 2
    # ------------------------------------------------------------------------
    print("\n" + "="*80)
    print("CONSTRUCTING LEVEL 2 META MODELS")
    print("="*80)
    
    # Meta-Model 1: LightGBM
    meta_lgb = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.02,
        num_leaves=15,
        random_state=GLOBAL_SEED,
        n_jobs=-1,
        verbose=-1
    )
    stacking_ensemble.add_level2_model("Meta_LGB", meta_lgb)
    
    # Meta-Model 2: XGBoost
    meta_xgb = xgb.XGBClassifier(
        n_estimators=500,
        learning_rate=0.02,
        max_depth=4,
        random_state=GLOBAL_SEED,
        n_jobs=-1,
        tree_method='gpu_hist' if torch.cuda.is_available() else 'hist'
    )
    stacking_ensemble.add_level2_model("Meta_XGB", meta_xgb)
    
    # Meta-Model 3: Logistic Regression
    meta_lr = LogisticRegression(
        C=1.0,
        random_state=GLOBAL_SEED,
        max_iter=1000,
        n_jobs=-1
    )
    stacking_ensemble.add_level2_model("Meta_LR", meta_lr)
    
    # Meta-Model 4: Ridge Classifier
    meta_ridge = RidgeClassifier(
        alpha=1.0,
        random_state=GLOBAL_SEED
    )
    stacking_ensemble.add_level2_model("Meta_Ridge", meta_ridge)
    
    # Train Level 2
    stacking_ensemble.train_level2(level1_oof_df, y_train, X_test_selected)
    
    # ------------------------------------------------------------------------
    # STAGE 10: WEIGHTED BLENDING OPTIMIZATION
    # ------------------------------------------------------------------------
    blending_optimizer = WeightedBlendingOptimizer()
    
    # Collect Level 2 OOF predictions
    level2_oof_list = [
        stacking_ensemble.level2_oof_predictions[name]
        for name, _ in stacking_ensemble.level2_models
    ]
    
    # Optimize weights
    optimal_weights, optimal_auc = blending_optimizer.optimize_weights(
        level2_oof_list, y_train
    )
    
    # Apply optimal weights to test predictions
    level2_test_list = [
        stacking_ensemble.test_predictions[f"{name}_L2"]
        for name, _ in stacking_ensemble.level2_models
    ]
    
    final_test_predictions = blending_optimizer.apply_weights(
        level2_test_list, optimal_weights
    )
    
    # ------------------------------------------------------------------------
    # STAGE 11: FINAL EVALUATION
    # ------------------------------------------------------------------------
    metric = CompetitionMetric()
    
    print("\n" + "="*80)
    print("FINAL PERFORMANCE SUMMARY")
    print("="*80)
    
    # Level 2 individual scores
    level2_scores = {}
    for name in stacking_ensemble.level2_oof_predictions.keys():
        score = metric.calculate_auc(
            y_train, stacking_ensemble.level2_oof_predictions[name]
        )
        level2_scores[name] = score
    
    # Optimized blend score
    blended_oof = blending_optimizer.apply_weights(level2_oof_list, optimal_weights)
    blend_score = metric.calculate_auc(y_train, blended_oof)
    level2_scores['Optimized_Blend'] = blend_score
    
    metric.print_performance_summary(level2_scores)
    
    # ROC Curve visualization
    metric.plot_roc_curve(y_train, blended_oof, 
                         title="Final Optimized Ensemble - ROC Curve")
    
    # ------------------------------------------------------------------------
    # STAGE 12: SUBMISSION FILE GENERATION
    # ------------------------------------------------------------------------
    submission[target_col] = final_test_predictions
    submission.to_csv('submission_elite_ensemble.csv', index=False)
    
    print("\n" + "="*80)
    print("SUBMISSION FILE GENERATED")
    print("="*80)
    print(f"\nFilename: submission_elite_ensemble.csv")
    print(f"Expected Public LB AUC: {blend_score:.4f}")
    print(f"Target Ranking: Top 3 (Medal Zone)")
    
    print("\nSubmission Preview:")
    print(submission.head(10).to_string(index=False))
    print("\n" + "="*80)
    print("PIPELINE EXECUTION COMPLETE")
    print("="*80)
    
    return submission, stacking_ensemble, optimal_weights

# ============================================================================
# SECTION 14: EXECUTION ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Execute main pipeline
    submission, ensemble, weights = execute_competition_pipeline()
    
    # Memory cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print("\nAll systems operational. Ready for submission.")

