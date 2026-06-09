import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

import os
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import pickle
from dataclasses import dataclass
from pathlib import Path

# Data Processing
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.metrics import roc_auc_score, log_loss, classification_report, confusion_matrix, accuracy_score
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import mutual_info_classif, RFECV, SelectKBest

# Machine Learning Models
import xgboost as xgb
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    GradientBoostingClassifier, HistGradientBoostingClassifier
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

# Ensemble
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.calibration import CalibratedClassifierCV

# Hyperparameter Optimization
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner

# Feature Engineering & Selection
from sklearn.feature_selection import mutual_info_classif
from itertools import combinations
import shap

from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
BAYESIAN_OPT_AVAILABLE = True

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class EliteConfigV2:
    """Elite configuration based on 2024-2025 research"""

    # File Paths
    TRAIN_PATH: str = '/kaggle/input/playground-series-s5e12/train.csv'
    TEST_PATH: str = '/kaggle/input/playground-series-s5e12/test.csv'
    SAMPLE_SUBMISSION_PATH: str = '/kaggle/input/playground-series-s5e12/sample_submission.csv'
    SUBMISSION_PATH: str = 'submission.csv'

    # Column Names
    TARGET: str = 'diagnosed_diabetes'
    ID_COL: str = 'id'

    # Categorical Features
    CATEGORICAL_FEATURES: List[str] = None

    def __post_init__(self):
        if self.CATEGORICAL_FEATURES is None:
            self.CATEGORICAL_FEATURES = [
                'gender', 'ethnicity', 'education_level', 'income_level',
                'smoking_status', 'employment_status'
            ]

    # Random Seed
    SEED: int = 42

    # Cross-Validation Strategy
    N_FOLDS: int = 2  # Optimal for large datasets

    # Feature Selection
    KEEP_TOP_FEATURES: int = 100  # Keep top features instead of aggressive filtering

    # Feature Selection (BorutaShap-inspired)
    USE_BORUTASHAP_SELECTION: bool = False
    FEATURE_SELECTION_THRESHOLD: float = 0.01
    MAX_FEATURES: int = 100  # Will test multiple sizes

    # Feature Selection - IMPROVED based on 2024 research
    USE_RFECV: bool = False  # RFECV is best per 2024 research
    MIN_FEATURES_TO_SELECT: int = 20  # Keep at least 30 features
    MAX_FEATURES: int = 50  # Maximum 50 features
    # SHAP-based selection (secondary)
    USE_SHAP_SELECTION: bool = False
    SHAP_THRESHOLD: float = 0.003  # Relaxed threshold

    # Feature Selection
    CORRELATION_THRESHOLD: float = 0.95  # Remove highly correlated features
    MIN_IMPORTANCE_THRESHOLD: float = 0.001  # Permutation importance threshold

    # Hyperparameter Tuning
    HPO_ITERATIONS: int = 30
    
    # Advanced Class Weight Optimization
    USE_FOCAL_LOSS: bool = True  # Use focal loss for XGBoost/LightGBM
    OPTIMIZE_CLASS_WEIGHTS: bool = True  # Use Bayesian optimization for class weights
    OPTIMIZE_THRESHOLD: bool = True  # Optimize decision threshold
    FAST_MODE: bool = True  # Enable fast mode to reduce computation time

    # Focal Loss Parameters (will be optimized)
    FOCAL_GAMMA: float = 2.0  # Focusing parameter
    FOCAL_ALPHA: float = 0.25  # Class balancing parameter


class AdvancedMedicalFeatureEngineer:
    """
    Advanced feature engineering based on medical research (2024-2025).

    References:
    - American Diabetes Association (ADA) Guidelines
    - Metabolic Syndrome (NCEP ATP III Criteria)
    - Framingham Risk Score
    """

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive medical features"""
        df = df.copy()

        # ═══════════════════════════════════════════════════════════════
        # LIPID PANEL RATIOS (Strongest CVD/Diabetes predictors)
        # ═══════════════════════════════════════════════════════════════

        if all(col in df.columns for col in ['cholesterol_total', 'hdl_cholesterol']):
            df['chol_hdl_ratio'] = df['cholesterol_total'] / (df['hdl_cholesterol'] + 0.01)
            df['non_hdl_cholesterol'] = df['cholesterol_total'] - df['hdl_cholesterol']

        if all(col in df.columns for col in ['ldl_cholesterol', 'hdl_cholesterol']):
            df['ldl_hdl_ratio'] = df['ldl_cholesterol'] / (df['hdl_cholesterol'] + 0.01)

        if all(col in df.columns for col in ['triglycerides', 'hdl_cholesterol']):
            # TG/HDL ratio - STRONGEST insulin resistance marker
            df['trig_hdl_ratio'] = df['triglycerides'] / (df['hdl_cholesterol'] + 0.01)
            df['trig_hdl_high_risk'] = (df['trig_hdl_ratio'] >= 3.0).astype(int)

        # ═══════════════════════════════════════════════════════════════
        # BLOOD PRESSURE METRICS
        # ═══════════════════════════════════════════════════════════════

        if all(col in df.columns for col in ['systolic_bp', 'diastolic_bp']):
            df['pulse_pressure'] = df['systolic_bp'] - df['diastolic_bp']
            df['mean_arterial_pressure'] = df['diastolic_bp'] + df['pulse_pressure'] / 3
            df['pp_map_ratio'] = df['pulse_pressure'] / (df['mean_arterial_pressure'] + 0.01)

        # ═══════════════════════════════════════════════════════════════
        # METABOLIC SYNDROME SCORE (Clinical Gold Standard)
        # ═══════════════════════════════════════════════════════════════

        metabolic_score = 0

        if 'bmi' in df.columns:
            df['is_obese'] = (df['bmi'] >= 30).astype(int)
            df['is_overweight'] = (df['bmi'] >= 25).astype(int)
            df['bmi_squared'] = df['bmi'] ** 2
            metabolic_score += df['is_obese']

        if all(col in df.columns for col in ['systolic_bp', 'diastolic_bp']):
            df['has_hypertension'] = ((df['systolic_bp'] >= 130) |
                                      (df['diastolic_bp'] >= 85)).astype(int)
            metabolic_score += df['has_hypertension']

        if 'triglycerides' in df.columns:
            df['high_triglycerides'] = (df['triglycerides'] >= 150).astype(int)
            metabolic_score += df['high_triglycerides']

        if 'hdl_cholesterol' in df.columns:
            df['low_hdl'] = (df['hdl_cholesterol'] < 40).astype(int)
            metabolic_score += df['low_hdl']

        df['metabolic_syndrome_score'] = metabolic_score
        df['has_metabolic_syndrome'] = (metabolic_score >= 3).astype(int)

        # ═══════════════════════════════════════════════════════════════
        # AGE-RISK INTERACTIONS (Critical for diabetes)
        # ═══════════════════════════════════════════════════════════════

        if 'age' in df.columns:
            df['age_high_risk'] = (df['age'] >= 45).astype(int)
            df['age_very_high_risk'] = (df['age'] >= 65).astype(int)
            df['age_squared'] = df['age'] ** 2
            df['age_cubed'] = df['age'] ** 3

            # Key age interactions
            if 'bmi' in df.columns:
                df['age_bmi'] = df['age'] * df['bmi']
                df['age_bmi_squared'] = df['age'] * df['bmi_squared']

            if 'family_history_diabetes' in df.columns:
                df['age_family_risk'] = df['age'] * df['family_history_diabetes']

            if 'metabolic_syndrome_score' in df.columns:
                df['age_metabolic'] = df['age'] * df['metabolic_syndrome_score']

        # ═══════════════════════════════════════════════════════════════
        # LIFESTYLE RISK FACTORS
        # ═══════════════════════════════════════════════════════════════

        if 'physical_activity_minutes_per_week' in df.columns:
            df['sedentary'] = (df['physical_activity_minutes_per_week'] < 150).astype(int)
            df['highly_active'] = (df['physical_activity_minutes_per_week'] >= 300).astype(int)
            df['activity_log'] = np.log1p(df['physical_activity_minutes_per_week'])

            # Activity-obesity interaction
            if 'bmi' in df.columns:
                df['sedentary_obese'] = df['sedentary'] * df['is_obese']

        # ═══════════════════════════════════════════════════════════════
        # COMPOSITE RISK SCORES
        # ═══════════════════════════════════════════════════════════════

        # Cardiovascular risk score
        cv_risk = 0
        if 'is_obese' in df.columns:
            cv_risk += df['is_obese'] * 2
        if 'has_hypertension' in df.columns:
            cv_risk += df['has_hypertension'] * 2
        if 'high_triglycerides' in df.columns:
            cv_risk += df['high_triglycerides']
        if 'low_hdl' in df.columns:
            cv_risk += df['low_hdl']

        df['cv_risk_score'] = cv_risk

        # Insulin resistance proxy
        ir_score = 0
        if 'trig_hdl_high_risk' in df.columns:
            ir_score += df['trig_hdl_high_risk'] * 3
        if 'is_obese' in df.columns:
            ir_score += df['is_obese'] * 2
        if 'sedentary' in df.columns:
            ir_score += df['sedentary']

        df['insulin_resistance_score'] = ir_score

        # ═══════════════════════════════════════════════════════════════
        # POLYNOMIAL & LOG TRANSFORMS (Top predictors only)
        # ═══════════════════════════════════════════════════════════════

        key_features = ['bmi', 'age', 'triglycerides', 'hdl_cholesterol', 'systolic_bp']
        for feat in key_features:
            if feat in df.columns:
                df[f'{feat}_log'] = np.log1p(np.abs(df[feat]))
                df[f'{feat}_sqrt'] = np.sqrt(np.abs(df[feat]))

        return df


class KFoldTargetEncoder:
    """
    K-Fold target encoding to prevent data leakage.
    Based on 2024 research: prevents overfitting better than label encoding.
    """

    def __init__(self, categorical_features: List[str], n_splits: int = 5,
                 smoothing: float = 1.0, seed: int = 42):
        self.categorical_features = categorical_features
        self.n_splits = n_splits
        self.smoothing = smoothing
        self.seed = seed
        self.global_means = {}
        self.encodings = {}

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Fit and transform training data using K-Fold"""
        X_encoded = X.copy()

        for col in self.categorical_features:
            if col not in X.columns:
                continue

            # Calculate global mean for this feature
            self.global_means[col] = y.mean()

            # Create K-Fold encoding - initialize with zeros
            target_enc_values = np.zeros(len(X))

            kf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)

            for train_idx, val_idx in kf.split(X, y):
                # Calculate means on training fold
                train_stats = y.iloc[train_idx].groupby(X[col].iloc[train_idx]).agg(['mean', 'count'])

                # Apply smoothing
                smoothed_means = (
                    (train_stats['mean'] * train_stats['count'] +
                     self.global_means[col] * self.smoothing) /
                    (train_stats['count'] + self.smoothing)
                )

                # Encode validation fold
                target_enc_values[val_idx] = (
                    X[col].iloc[val_idx].map(smoothed_means).fillna(self.global_means[col]).values
                )

            # Add encoded column
            X_encoded[f'{col}_target_enc'] = target_enc_values

            # Store final encodings for transform
            final_stats = y.groupby(X[col]).agg(['mean', 'count'])
            self.encodings[col] = (
                (final_stats['mean'] * final_stats['count'] +
                 self.global_means[col] * self.smoothing) /
                (final_stats['count'] + self.smoothing)
            )

        return X_encoded

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform test data using fitted encodings"""
        X_encoded = X.copy()

        for col in self.categorical_features:
            if col not in X.columns or col not in self.encodings:
                continue

            X_encoded[f'{col}_target_enc'] = (
                X[col].map(self.encodings[col]).fillna(self.global_means[col])
            )

        return X_encoded


class ShapFeatureSelector:
    """
    SHAP-based feature selection inspired by BorutaShap.
    """

    def __init__(self, n_estimators: int = 100, threshold: float = 0.001, max_features: int = 50):
        self.n_estimators = n_estimators
        self.threshold = threshold
        self.max_features = max_features
        self.selected_features = None
        self.feature_importances = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fit selector using SHAP values"""
        print("\n  → Running SHAP-based feature selection...")

        # Train a fast model for SHAP
        model = LGBMClassifier(
            n_estimators=self.n_estimators,
            learning_rate=0.05,
            max_depth=5,
            random_state=42,
            verbose=-1,
            n_jobs=-1
        )
        model.fit(X, y)

        # Calculate SHAP values
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        # Get mean absolute SHAP values
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # For binary classification

        mean_abs_shap = np.abs(shap_values).mean(axis=0)

        # Create feature importance dataframe
        self.feature_importances = pd.DataFrame({
            'feature': X.columns,
            'shap_importance': mean_abs_shap,
            'tree_importance': model.feature_importances_
        })

        # Combined importance score
        self.feature_importances['combined_importance'] = (
            0.6 * self.feature_importances['shap_importance'] / self.feature_importances['shap_importance'].max() +
            0.4 * self.feature_importances['tree_importance'] / self.feature_importances['tree_importance'].max()
        )

        self.feature_importances = self.feature_importances.sort_values(
            'combined_importance', ascending=False
        ).reset_index(drop=True)

        # Select features above threshold
        selected = self.feature_importances[
            self.feature_importances['combined_importance'] >= self.threshold
        ]['feature'].tolist()

        # Limit to max_features
        self.selected_features = selected[:self.max_features]

        print(f"    ✓ Selected {len(self.selected_features)} features out of {len(X.columns)}")
        print(f"\n    Top 15 features:")
        for idx, row in self.feature_importances.head(15).iterrows():
            print(f"      {idx+1:2d}. {row['feature']:40s} | Score: {row['combined_importance']:.4f}")

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data to selected features"""
        if self.selected_features is None:
            raise ValueError("Must fit selector before transform")

        return X[self.selected_features]

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Fit and transform in one step"""
        self.fit(X, y)
        return self.transform(X)


class SmartFeatureSelector:
    """Select top features by mutual information - keeps important interactions"""

    def __init__(self, top_k=100, seed=42):
        self.top_k = top_k
        self.seed = seed
        self.selected_features = None

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Select top K features by mutual information"""
        print(f"\n{'='*70}")
        print("SMART FEATURE SELECTION")
        print(f"{'='*70}")
        print(f"  Starting features: {X.shape[1]}")

        # Calculate mutual information
        mi_scores = mutual_info_classif(X, y, random_state=self.seed)

        # Rank features
        mi_df = pd.DataFrame({
            'feature': X.columns,
            'mi_score': mi_scores
        }).sort_values('mi_score', ascending=False)

        # Select top K
        self.selected_features = mi_df.head(self.top_k)['feature'].tolist()

        print(f"\n  Top 20 Features by Mutual Information:")
        for idx, row in mi_df.head(20).iterrows():
            print(f"    {row['feature']:45s} {row['mi_score']:.6f}")

        print(f"\n  Selected features: {len(self.selected_features)}")
        print(f"{'='*70}\n")

        return X[self.selected_features]

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform using selected features"""
        if self.selected_features is None:
            raise ValueError("Must fit before transform")
        return X[self.selected_features]


class RFECVFeatureSelector:
    """
    RFECV-based feature selection.

    Based on 2024 research: "RFECV proved to be the most effective method"
    Source: Frontiers in Artificial Intelligence, 2024
    """

    def __init__(self, min_features: int = 30, cv: int = 5, seed: int = 42):
        self.min_features = min_features
        self.cv = cv
        self.seed = seed
        self.selected_features = None
        self.selector = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fit RFECV selector"""
        print("\n" + "="*70)
        print("RFECV FEATURE SELECTION (2024 Best Method)")
        print("="*70)
        print(f"  Starting features: {X.shape[1]}")
        print(f"  Minimum features: {self.min_features}")

        # Use LightGBM as estimator (fast and accurate)
        estimator = LGBMClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=5,
            random_state=self.seed,
            verbose=-1,
            n_jobs=-1
        )

        # RFECV with stratified CV
        self.selector = RFECV(
            estimator=estimator,
            step=1,
            cv=StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.seed),
            scoring='roc_auc',
            min_features_to_select=self.min_features,
            n_jobs=-1
        )

        print(f"\n  Running RFECV with {self.cv}-fold CV...")
        self.selector.fit(X, y)

        # Get selected features
        self.selected_features = X.columns[self.selector.support_].tolist()

        print(f"\n  ✓ RFECV complete!")
        print(f"  Selected features: {len(self.selected_features)}")
        print(f"  Optimal score: {self.selector.cv_results_['mean_test_score'].max():.6f}")

        # Show top features by ranking
        feature_ranking = pd.DataFrame({
            'feature': X.columns,
            'ranking': self.selector.ranking_,
            'selected': self.selector.support_
        }).sort_values('ranking')

        print(f"\n  Top 20 Selected Features:")
        for idx, row in feature_ranking[feature_ranking['selected']].head(20).iterrows():
            print(f"    {row['feature']}")

        print("="*70 + "\n")

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform to selected features"""
        if self.selected_features is None:
            raise ValueError("Must fit before transform")
        return X[self.selected_features]

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Fit and transform"""
        self.fit(X, y)
        return self.transform(X)


class HybridFeatureSelector:
    """
    Hybrid approach: RFECV + SHAP validation
    Best of both worlds per 2024 research
    """

    def __init__(self, min_features: int = 30, max_features: int = 50,
                 shap_threshold: float = 0.003, cv: int = 5, seed: int = 42):
        self.min_features = min_features
        self.max_features = max_features
        self.shap_threshold = shap_threshold
        self.cv = cv
        self.seed = seed
        self.selected_features = None
        self.feature_importances = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fit hybrid selector"""
        print("\n" + "="*70)
        print("HYBRID FEATURE SELECTION (RFECV + SHAP)")
        print("="*70)

        # Step 1: RFECV to get candidate features
        rfecv_selector = RFECVFeatureSelector(
            min_features=self.min_features,
            cv=self.cv,
            seed=self.seed
        )
        X_rfecv = rfecv_selector.fit_transform(X, y)
        rfecv_features = rfecv_selector.selected_features

        print(f"\n  RFECV selected {len(rfecv_features)} features")

        # Step 2: SHAP validation on RFECV features
        print(f"\n  Running SHAP validation...")

        model = LGBMClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=5,
            random_state=self.seed,
            verbose=-1,
            n_jobs=-1
        )
        model.fit(X_rfecv, y)

        # Calculate SHAP values
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_rfecv)

        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        mean_abs_shap = np.abs(shap_values).mean(axis=0)

        # Create importance dataframe
        self.feature_importances = pd.DataFrame({
            'feature': X_rfecv.columns,
            'shap_importance': mean_abs_shap,
            'tree_importance': model.feature_importances_
        })

        # Normalize and combine
        self.feature_importances['shap_norm'] = (
            self.feature_importances['shap_importance'] /
            self.feature_importances['shap_importance'].max()
        )
        self.feature_importances['tree_norm'] = (
            self.feature_importances['tree_importance'] /
            self.feature_importances['tree_importance'].max()
        )
        self.feature_importances['combined_score'] = (
            0.6 * self.feature_importances['shap_norm'] +
            0.4 * self.feature_importances['tree_norm']
        )

        self.feature_importances = self.feature_importances.sort_values(
            'combined_score', ascending=False
        ).reset_index(drop=True)

        # Select features above threshold, but limit to max_features
        selected = self.feature_importances[
            self.feature_importances['combined_score'] >= self.shap_threshold
        ]['feature'].tolist()

        self.selected_features = selected[:self.max_features]

        print(f"\n  ✓ Hybrid selection complete!")
        print(f"  Final selected features: {len(self.selected_features)}")

        print(f"\n  Top 15 Features by Combined Score:")
        for idx, row in self.feature_importances.head(15).iterrows():
            print(f"    {idx+1:2d}. {row['feature']:45s} | {row['combined_score']:.4f}")

        print("="*70 + "\n")

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform to selected features"""
        if self.selected_features is None:
            raise ValueError("Must fit before transform")
        return X[self.selected_features]

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Fit and transform"""
        self.fit(X, y)
        return self.transform(X)


class FocalLoss:
    """
    Focal Loss implementation for XGBoost and LightGBM.

    Reference: Lin et al. (2017) "Focal Loss for Dense Object Detection"
    Adapted for gradient boosting as per "Imbalance-XGBoost" (2019)

    Focal Loss = -α * (1-p)^γ * log(p) for positive class
                 -(1-α) * p^γ * log(1-p) for negative class

    Parameters:
        gamma (float): Focusing parameter (γ). Higher values focus more on hard examples.
                      Typical range: [0, 5]. Default: 2.0
        alpha (float): Class balancing parameter (α). Weight for positive class.
                      Typical range: [0.25, 0.75]. Default: 0.25
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        self.gamma = gamma
        self.alpha = alpha

    def focal_loss_lgb(self, y_true, y_pred):
        """
        Focal loss for LightGBM.
        Returns: (gradient, hessian)
        """
        # Clip predictions to avoid log(0)
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Compute focal weight
        p = 1.0 / (1.0 + np.exp(-y_pred))  # sigmoid

        # For positive class (y=1)
        alpha_factor = np.where(y_true == 1, self.alpha, 1 - self.alpha)
        focal_weight = np.where(
            y_true == 1,
            (1 - p) ** self.gamma,
            p ** self.gamma
        )

        # Gradient and Hessian for focal loss
        grad = focal_weight * alpha_factor * (p - y_true)
        hess = focal_weight * alpha_factor * p * (1 - p)

        return grad, hess

    def focal_loss_xgb(self, y_true, y_pred):
        """
        Focal loss for XGBoost (sklearn API).

        IMPORTANT: When used with XGBoostClassifier (sklearn API), the custom objective
        function receives arguments in order: (y_true, y_pred) where both are numpy arrays.
        This is different from the low-level xgboost.train() API.

        Returns: (gradient, hessian)
        """
        # Note: y_true and y_pred are already numpy arrays (not DMatrix)
        # XGBoostClassifier's sklearn wrapper handles the conversion

        # Clip predictions to avoid numerical issues
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Compute probabilities (sigmoid)
        p = 1.0 / (1.0 + np.exp(-y_pred))

        # Alpha factor for class balancing
        alpha_factor = np.where(y_true == 1, self.alpha, 1 - self.alpha)

        # Focal weight: (1-p)^gamma for positive, p^gamma for negative
        focal_weight = np.where(
            y_true == 1,
            (1 - p) ** self.gamma,
            p ** self.gamma
        )

        # Gradient and Hessian for focal loss
        grad = focal_weight * alpha_factor * (p - y_true)
        hess = focal_weight * alpha_factor * p * (1 - p) + 1e-6  # Add small epsilon for stability

        return grad, hess


class BayesianClassWeightOptimizer:
    """
    Optimize class weights using Bayesian optimization.

    Unlike simple 'balanced' approach (which uses inverse frequency),
    this finds optimal weights through systematic search to maximize AUC.

    Reference:
    - "Class imbalance learning with Bayesian optimization" (Nature 2022)
    - Research shows optimal weights often differ from inverse frequency
    """

    def __init__(self, n_iterations: int = 30, cv_folds: int = 3, seed: int = 42):
        self.n_iterations = n_iterations
        self.cv_folds = cv_folds
        self.seed = seed
        self.best_weights_ = None
        self.search_history_ = []

    def _objective_function(self, weight_ratio, model, X, y):
        """
        FAST objective function to maximize AUC with given weight ratio.

        OPTIMIZED: Uses single split + data sampling for speed.
        """
        # Set class weights
        class_weights = {0: 1.0, 1: weight_ratio}

        # Clone model and set weights
        if hasattr(model, 'class_weight'):
            model_clone = model.__class__(**model.get_params())
            model_clone.set_params(class_weight=class_weights)
        elif hasattr(model, 'scale_pos_weight'):  # XGBoost
            model_clone = model.__class__(**model.get_params())
            model_clone.set_params(scale_pos_weight=weight_ratio)
        else:
            return 0.0  # Model doesn't support class weights

        # FAST MODE: Single split instead of CV
        from sklearn.model_selection import train_test_split
        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=0.25, stratify=y, random_state=self.seed
        )

        model_clone.fit(X_tr, y_tr)
        y_pred = model_clone.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred)

        self.search_history_.append((weight_ratio, auc))

        return auc

    def optimize(self, model, X: pd.DataFrame, y: pd.Series,
                 search_range: Tuple[float, float] = (0.5, 5.0)) -> Dict[int, float]:
        """
        Find optimal class weights using FAST grid search.

        OPTIMIZED: Reduced search space and iterations for speed.

        Args:
            model: Scikit-learn compatible model
            X: Training features
            y: Training labels
            search_range: (min_weight, max_weight) for minority class

        Returns:
            Dictionary of optimal class weights {0: weight_0, 1: weight_1}
        """
        print(f"\n  Optimizing class weights for {model.__class__.__name__}...")
        print(f"    Search range: {search_range}")

        # REDUCED: Only coarse search, skip fine search for speed
        search_ratios = np.logspace(np.log10(search_range[0]),
                                    np.log10(search_range[1]),
                                    num=8)  # REDUCED from 10 to 8

        best_auc = 0
        best_ratio = 1.0

        for ratio in search_ratios:
            auc = self._objective_function(ratio, model, X, y)
            if auc > best_auc:
                best_auc = auc
                best_ratio = ratio

        # Store results
        self.best_weights_ = {0: 1.0, 1: best_ratio}

        inverse_freq_ratio = (y==0).sum()/(y==1).sum()
        print(f"    Best weight ratio: {best_ratio:.4f}")
        print(f"    Best AUC: {best_auc:.6f}")
        print(f"    vs. Inverse freq ({inverse_freq_ratio:.4f})")

        return self.best_weights_


class ThresholdOptimizer:
    """
    Optimize classification threshold for imbalanced data.

    Reference: "Balancing the Scales" (arXiv 2024) - shows threshold
    calibration is most consistently effective for imbalanced data.
    """

    def __init__(self, method: str = 'youden'):
        """
        Args:
            method: 'youden' (Youden's J), 'f1' (maximize F1), or 'gmean' (geometric mean)
        """
        self.method = method
        self.optimal_threshold_ = 0.5

    def optimize(self, y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """Find optimal threshold."""

        if self.method == 'youden':
            # Youden's J statistic: max(sensitivity + specificity - 1)
            from sklearn.metrics import roc_curve
            fpr, tpr, thresholds = roc_curve(y_true, y_prob)
            j_scores = tpr - fpr
            optimal_idx = np.argmax(j_scores)
            self.optimal_threshold_ = thresholds[optimal_idx]

        elif self.method == 'f1':
            # Maximize F1 score
            from sklearn.metrics import precision_recall_curve
            precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
            optimal_idx = np.argmax(f1_scores)
            self.optimal_threshold_ = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5

        elif self.method == 'gmean':
            # Geometric mean of sensitivity and specificity
            from sklearn.metrics import roc_curve
            fpr, tpr, thresholds = roc_curve(y_true, y_prob)
            gmean = np.sqrt(tpr * (1 - fpr))
            optimal_idx = np.argmax(gmean)
            self.optimal_threshold_ = thresholds[optimal_idx]

        return self.optimal_threshold_

    def predict(self, y_prob: np.ndarray) -> np.ndarray:
        """Make predictions using optimal threshold."""
        return (y_prob >= self.optimal_threshold_).astype(int)


# ═══════════════════════════════════════════════════════════════════════════
# MULTI-LAYER STACKING ENSEMBLE (AutoGluon Approach)
# ═══════════════════════════════════════════════════════════════════════════

class OptimizedEnsemble:
    """
    Advanced 2-layer ensemble with optimized class weights and focal loss.

    Improvements over V1:
    1. Focal loss for gradient boosting models
    2. Bayesian optimization for class weights
    3. Individual weight optimization per model
    4. Threshold optimization for final predictions

    Based on 2024-2025 research on imbalanced classification.
    """

    def __init__(self, config: EliteConfigV2):
        self.config = config
        self.base_models = {}
        self.meta_model = None
        self.threshold_optimizer = None
        self.optimal_weights = {}  # Store optimal weights per model
        self.focal_loss = None

    def optimize_focal_loss_params(self, X: pd.DataFrame, y: pd.Series) -> Tuple[float, float]:
        """
        Optimize focal loss parameters (gamma, alpha) using FAST grid search.

        OPTIMIZED: Uses only 1 fold + subset of data for speed.

        Returns:
            (optimal_gamma, optimal_alpha)
        """
        print(f"\n  Optimizing Focal Loss Parameters (fast mode)...")

        # REDUCED search space for speed
        gamma_range = [1.0, 2.0, 3.0]  # Reduced from 5 to 3 values
        alpha_range = [0.25, 0.5]       # Reduced from 3 to 2 values

        best_auc = 0
        best_gamma = 2.0
        best_alpha = 0.25

        # FAST MODE: Single train/val split instead of 3-fold CV
        # Use stratified sampling for representative split
        from sklearn.model_selection import train_test_split
        X_sample, _, y_sample, _ = train_test_split(
            X, y, train_size=0.3, stratify=y, random_state=self.config.SEED
        )

        X_tr, X_val, y_tr, y_val = train_test_split(
            X_sample, y_sample, test_size=0.33, stratify=y_sample, random_state=self.config.SEED
        )

        for gamma in gamma_range:
            for alpha in alpha_range:
                # Test with LightGBM (faster than XGBoost)
                model = LGBMClassifier(
                    n_estimators=50,  # REDUCED from 100 for speed
                    learning_rate=0.1,  # INCREASED for faster convergence
                    random_state=self.config.SEED,
                    verbose=-1
                )

                model.fit(X_tr, y_tr)
                y_pred = model.predict_proba(X_val)[:, 1]
                auc = roc_auc_score(y_val, y_pred)

                if auc > best_auc:
                    best_auc = auc
                    best_gamma = gamma
                    best_alpha = alpha

        print(f"    Optimal gamma: {best_gamma}, alpha: {best_alpha}")
        print(f"    Best AUC: {best_auc:.6f}")

        return best_gamma, best_alpha

    def create_tuned_models(self, X: pd.DataFrame = None, y: pd.Series = None) -> Dict[str, Any]:
        """
        Create models with OPTIMIZED hyperparameters and class weights.

        If X and y provided, will optimize class weights for each model.
        """
        models = {}

        # ═══════════════════════════════════════════════════════════════
        # GRADIENT BOOSTING MODELS with Focal Loss (Best for tabular data)
        # ═══════════════════════════════════════════════════════════════

        if self.config.USE_FOCAL_LOSS and X is not None and y is not None and not self.config.FAST_MODE:
            # Optimize focal loss parameters (only if FAST_MODE is disabled)
            optimal_gamma, optimal_alpha = self.optimize_focal_loss_params(X, y)
            self.focal_loss = FocalLoss(gamma=optimal_gamma, alpha=optimal_alpha)
        else:
            # Use default parameters (faster)
            self.focal_loss = FocalLoss(gamma=self.config.FOCAL_GAMMA, alpha=self.config.FOCAL_ALPHA)
            if self.config.USE_FOCAL_LOSS and self.config.FAST_MODE:
                print(f"\n  Using default Focal Loss params (gamma={self.config.FOCAL_GAMMA}, alpha={self.config.FOCAL_ALPHA}) - FAST MODE")

        # XGBoost with Focal Loss
        if self.config.USE_FOCAL_LOSS:
            models['xgb'] = xgb.XGBClassifier(
                n_estimators=1000,
                learning_rate=0.03,
                max_depth=6,
                min_child_weight=3,
                subsample=0.8,
                colsample_bytree=0.8,
                gamma=1.0,
                reg_alpha=0.5,
                reg_lambda=1.0,
                random_state=self.config.SEED,
                tree_method='hist',
                eval_metric='auc',
                objective=self.focal_loss.focal_loss_xgb,
                disable_default_eval_metric=True
            )
        else:
            models['xgb'] = xgb.XGBClassifier(
                n_estimators=1000,
                learning_rate=0.03,
                max_depth=6,
                min_child_weight=3,
                subsample=0.8,
                colsample_bytree=0.8,
                gamma=1.0,
                reg_alpha=0.5,
                reg_lambda=1.0,
                scale_pos_weight=1.0,  # Will be optimized
                random_state=self.config.SEED,
                tree_method='hist',
                eval_metric='auc'
            )

        # LightGBM with Focal Loss
        if self.config.USE_FOCAL_LOSS:
            models['lgbm'] = LGBMClassifier(
                n_estimators=1000,
                learning_rate=0.03,
                num_leaves=31,
                max_depth=6,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.5,
                reg_lambda=1.0,
                random_state=self.config.SEED,
                verbose=-1,
                objective=self.focal_loss.focal_loss_lgb
            )
        else:
            models['lgbm'] = LGBMClassifier(
                n_estimators=1000,
                learning_rate=0.03,
                num_leaves=31,
                max_depth=6,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.5,
                reg_lambda=1.0,
                random_state=self.config.SEED,
                verbose=-1,
                class_weight='balanced',
                is_unbalance=True
            )

        # CatBoost - Best at handling categoricals
        models['catboost'] = CatBoostClassifier(
            iterations=1000,
            learning_rate=0.03,
            depth=6,
            l2_leaf_reg=3.0,
            random_state=self.config.SEED,
            verbose=False,
            auto_class_weights='Balanced'
        )

        # HistGradientBoosting - Fast sklearn implementation
        models['hist_gb'] = HistGradientBoostingClassifier(
            max_iter=1000,
            learning_rate=0.03,
            max_depth=6,
            min_samples_leaf=20,
            l2_regularization=1.0,
            random_state=self.config.SEED,
            early_stopping=False
        )

        # ═══════════════════════════════════════════════════════════════
        # TREE ENSEMBLES (Diversity)
        # ═══════════════════════════════════════════════════════════════

        models['rf'] = RandomForestClassifier(
            n_estimators=500,
            max_depth=12,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            random_state=self.config.SEED,
            n_jobs=-1,
            class_weight='balanced'
        )

        models['et'] = ExtraTreesClassifier(
            n_estimators=500,
            max_depth=12,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            random_state=self.config.SEED,
            n_jobs=-1,
            class_weight='balanced'
        )

        # ═══════════════════════════════════════════════════════════════
        # LINEAR MODEL (Calibrated)
        # ═══════════════════════════════════════════════════════════════

        # FIXED: Proper solver for elasticnet
        models['lr'] = LogisticRegression(
            C=0.1,
            penalty='elasticnet',
            solver='saga',  # REQUIRED for elasticnet
            l1_ratio=0.5,
            max_iter=1000,
            random_state=self.config.SEED,
            class_weight='balanced',
            n_jobs=-1
        )

        return models

    def train(self, X: pd.DataFrame, y: pd.Series, X_test: pd.DataFrame):
        """
        Train 2-layer ensemble with ADVANCED class weight optimization.

        New features:
        1. Focal loss optimization for gradient boosting
        2. Bayesian class weight optimization per model
        3. Threshold calibration for final predictions
        """

        print(f"\n{'='*70}")
        print("ADVANCED ENSEMBLE TRAINING WITH OPTIMIZED CLASS WEIGHTS")
        print(f"{'='*70}")

        # ═══════════════════════════════════════════════════════════════
        # STEP 0: Optimize Class Weights (if enabled)
        # ═══════════════════════════════════════════════════════════════

        if self.config.OPTIMIZE_CLASS_WEIGHTS:
            print(f"\n[STEP 0] Bayesian Class Weight Optimization")
            print(f"{'='*70}")

        # ═══════════════════════════════════════════════════════════════
        # LAYER 1: Diverse Base Models with Optimized Weights
        # ═══════════════════════════════════════════════════════════════

        print(f"\n[LAYER 1] Training Base Models ({self.config.N_FOLDS}-Fold CV)")

        # Create models with focal loss optimization
        base_models = self.create_tuned_models(X, y)

        # Calculate class balance info
        neg_count = (y == 0).sum()
        pos_count = (y == 1).sum()
        inverse_ratio = neg_count / pos_count

        print(f"\n  Class balance: {neg_count} negative, {pos_count} positive")
        print(f"  Inverse frequency ratio: {inverse_ratio:.3f}")

        # Optimize class weights for each model individually
        if self.config.OPTIMIZE_CLASS_WEIGHTS and not self.config.FAST_MODE:
            # FULL OPTIMIZATION (slower but potentially better)
            weight_optimizer = BayesianClassWeightOptimizer(
                n_iterations=20,
                cv_folds=3,
                seed=self.config.SEED
            )

            for model_name, model in base_models.items():
                # Skip models with focal loss (already optimized)
                if self.config.USE_FOCAL_LOSS and model_name in ['xgb', 'lgbm']:
                    print(f"\n  {model_name.upper()}: Using Focal Loss (skipping weight optimization)")
                    continue

                # Optimize weights
                optimal_weights = weight_optimizer.optimize(model, X, y)
                self.optimal_weights[model_name] = optimal_weights

                # Apply optimized weights
                if model_name == 'xgb' and not self.config.USE_FOCAL_LOSS:
                    model.set_params(scale_pos_weight=optimal_weights[1])
                elif hasattr(model, 'class_weight'):
                    model.set_params(class_weight=optimal_weights)

        elif self.config.OPTIMIZE_CLASS_WEIGHTS and self.config.FAST_MODE:
            # FAST MODE: Skip weight optimization entirely (use built-in 'balanced')
            print(f"\n  FAST MODE: Using default 'balanced' weights for all models")
            # All models already have class_weight='balanced' or auto_class_weights='Balanced'

        else:
            # Use default balanced weights
            print(f"\n  Using default 'balanced' class weights for all models")
            for model_name, model in base_models.items():
                if model_name == 'xgb' and not self.config.USE_FOCAL_LOSS:
                    model.set_params(scale_pos_weight=inverse_ratio)

        # Out-of-fold predictions
        oof_preds = np.zeros((len(X), len(base_models)))
        test_preds = np.zeros((len(X_test), len(base_models)))

        skf = StratifiedKFold(n_splits=self.config.N_FOLDS, shuffle=True,
                             random_state=self.config.SEED)

        for model_idx, (model_name, model) in enumerate(base_models.items()):
            print(f"\n  Training {model_name.upper()}...")

            fold_scores = []
            test_pred_folds = []

            for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
                X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

                # Clone model
                model_fold = model.__class__(**model.get_params())

                # Fit model
                model_fold.fit(X_tr, y_tr)

                # OOF predictions - handle custom objectives
                if self.config.USE_FOCAL_LOSS and model_name in ['xgb', 'lgbm']:
                    # Models with custom objectives return raw predictions
                    # Need to apply sigmoid for probabilities
                    raw_pred_val = model_fold.predict(X_val)
                    oof_preds[val_idx, model_idx] = 1.0 / (1.0 + np.exp(-raw_pred_val))
                else:
                    # Standard models with predict_proba
                    oof_preds[val_idx, model_idx] = model_fold.predict_proba(X_val)[:, 1]

                # Test predictions - handle custom objectives
                if self.config.USE_FOCAL_LOSS and model_name in ['xgb', 'lgbm']:
                    raw_pred_test = model_fold.predict(X_test)
                    test_pred_folds.append(1.0 / (1.0 + np.exp(-raw_pred_test)))
                else:
                    test_pred_folds.append(model_fold.predict_proba(X_test)[:, 1])

                # Fold score
                fold_score = roc_auc_score(y_val, oof_preds[val_idx, model_idx])
                fold_scores.append(fold_score)

            # Average test predictions
            test_preds[:, model_idx] = np.mean(test_pred_folds, axis=0)

            # Overall OOF score
            oof_score = roc_auc_score(y, oof_preds[:, model_idx])
            print(f"    OOF AUC: {oof_score:.6f} (±{np.std(fold_scores):.6f})")

            # Store model
            self.base_models[model_name] = model

        layer1_auc = roc_auc_score(y, oof_preds.mean(axis=1))
        print(f"\n  Layer 1 Ensemble AUC: {layer1_auc:.6f}")

        # ═══════════════════════════════════════════════════════════════
        # LAYER 2: Meta Model with Calibration
        # ═══════════════════════════════════════════════════════════════

        print(f"\n[LAYER 2] Training Calibrated Meta Model")

        # Use LogisticRegression as meta model with calibration
        base_meta = LogisticRegression(
            C=1.0,
            solver='lbfgs',
            max_iter=1000,
            random_state=self.config.SEED,
            class_weight='balanced'
        )

        # Calibrate with isotonic regression (best for ensemble outputs)
        self.meta_model = CalibratedClassifierCV(
            base_meta,
            method='isotonic',
            cv=5
        )

        self.meta_model.fit(oof_preds, y)

        # Final predictions
        final_oof = self.meta_model.predict_proba(oof_preds)[:, 1]
        final_test = self.meta_model.predict_proba(test_preds)[:, 1]

        final_auc = roc_auc_score(y, final_oof)

        print(f"\n  Calibrated AUC (before threshold opt): {final_auc:.6f}")

        # ═══════════════════════════════════════════════════════════════
        # STEP 3: Threshold Optimization (if enabled)
        # ═══════════════════════════════════════════════════════════════

        if self.config.OPTIMIZE_THRESHOLD:
            print(f"\n[STEP 3] Threshold Optimization")
            print(f"{'='*70}")

            self.threshold_optimizer = ThresholdOptimizer(method='youden')
            optimal_threshold = self.threshold_optimizer.optimize(y, final_oof)

            print(f"  Optimal threshold: {optimal_threshold:.4f} (default: 0.5)")

            # Evaluate with optimized threshold
            from sklearn.metrics import classification_report, confusion_matrix
            y_pred_optimized = self.threshold_optimizer.predict(final_oof)

            print(f"\n  Performance with Optimized Threshold:")
            print(f"    Confusion Matrix:")
            cm = confusion_matrix(y, y_pred_optimized)
            print(f"      TN={cm[0,0]}, FP={cm[0,1]}")
            print(f"      FN={cm[1,0]}, TP={cm[1,1]}")

            from sklearn.metrics import precision_score, recall_score, f1_score
            precision = precision_score(y, y_pred_optimized)
            recall = recall_score(y, y_pred_optimized)
            f1 = f1_score(y, y_pred_optimized)

            print(f"    Precision: {precision:.4f}")
            print(f"    Recall:    {recall:.4f}")
            print(f"    F1 Score:  {f1:.4f}")

        print(f"\n  FINAL CALIBRATED AUC: {final_auc:.6f}")
        print(f"  Improvement over Layer 1: {(final_auc - layer1_auc):.6f}")
        print(f"{'='*70}\n")

        return final_oof, final_test


config = EliteConfigV2()
np.random.seed(config.SEED)

print(f"Configuration:")
print(f"  → Seed: {config.SEED}")
print(f"  → CV Folds: {config.N_FOLDS}")
print(f"  → Top Features: {config.KEEP_TOP_FEATURES}")
print(f"  → BORUTASHAP: {config.USE_BORUTASHAP_SELECTION}")
print(f"  → RFECV: {config.USE_RFECV}")

print(f"\n  Advanced Class Weight Optimization:")
print(f"    ✓ Focal Loss: {'ENABLED' if config.USE_FOCAL_LOSS else 'DISABLED'}")
print(f"    ✓ Bayesian Weight Optimization: {'ENABLED' if config.OPTIMIZE_CLASS_WEIGHTS else 'DISABLED'}")
print(f"    ✓ Threshold Calibration: {'ENABLED' if config.OPTIMIZE_THRESHOLD else 'DISABLED'}")
print(f"  → Calibrated probabilities enabled")



train_df = pd.read_csv(config.TRAIN_PATH)
test_df = pd.read_csv(config.TEST_PATH)

print(f"  Train shape: {train_df.shape}")
print(f"  Test shape: {test_df.shape}")

target_dist = train_df[config.TARGET].value_counts(normalize=True)
print(f"  Target: Class 0={target_dist[0]:.2%}, Class 1={target_dist[1]:.2%}")

# Save IDs
test_ids = test_df[config.ID_COL].copy()
train_df = train_df.drop(columns=[config.ID_COL])
test_df = test_df.drop(columns=[config.ID_COL])

y = train_df[config.TARGET].copy()
X = train_df.drop(columns=[config.TARGET])
X_test = test_df.copy()


# Label encode categoricals
for col in config.CATEGORICAL_FEATURES:
    if col in X.columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        X_test[col] = X_test[col].astype(str).apply(
            lambda x: le.transform([x])[0] if x in le.classes_ else -1
        )

# Handle missing values - fit ONLY on train
train_median = X.median()
X = X.fillna(train_median)
X_test = X_test.fillna(train_median)

print(f"  ✓ Preprocessing complete")


engineer = AdvancedMedicalFeatureEngineer()

#print(f"  Original features: {X.shape[1]}")
#X = engineer.create_features(X)
#X_test = engineer.create_features(X_test)

# Ensure same columns
missing_cols = set(X.columns) - set(X_test.columns)
for col in missing_cols:
    X_test[col] = 0
X_test = X_test[X.columns]

# Handle any NaNs from feature engineering
X_train = X.fillna(0)
X_test = X_test.fillna(0)

print(f"  → Final feature count: {X_train.shape[1]}")


y_train = train_df[config.TARGET].copy()

target_encoder = KFoldTargetEncoder(
    categorical_features=config.CATEGORICAL_FEATURES,
    n_splits=config.N_FOLDS,
    seed=config.SEED
)

X_train = target_encoder.fit_transform(X_train, y_train)
X_test = target_encoder.transform(X_test)

print(f"  ✓ Target encoding complete")
print(f"  → Features after encoding: {X_train.shape[1]}")


scaler = StandardScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=X_train.columns,
    index=X_train.index
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),
    columns=X_test.columns,
    index=X_test.index
)


if config.USE_BORUTASHAP_SELECTION:
    
    selector = ShapFeatureSelector(
    n_estimators=100,
    threshold=config.FEATURE_SELECTION_THRESHOLD,
    max_features=config.MAX_FEATURES
    )
    
    X_train_selected = selector.fit_transform(X_train_scaled, y_train)
    X_test_selected = selector.transform(X_test_scaled)
    
    print(f"  ✓ Feature selection complete")
    print(f"  → Selected features: {X_train_selected.shape[1]}")
elif config.USE_RFECV:
    selector = HybridFeatureSelector(
    min_features=config.MIN_FEATURES_TO_SELECT,
    max_features=config.MAX_FEATURES,
    shap_threshold=config.SHAP_THRESHOLD,
    cv=config.N_FOLDS,
    seed=config.SEED
    )

    X_train_selected = selector.fit_transform(X_train_scaled, y_train)
    X_test_selected = selector.transform(X_test_scaled)
else:
    X_train_selected = X_train_scaled
    X_test_selected = X_test_scaled


ensemble = OptimizedEnsemble(config)
final_oof, final_test = ensemble.train(X_train_selected, y_train, X_test_selected)


submission = pd.DataFrame({
    config.ID_COL: test_ids,
    config.TARGET: final_test
})

submission.to_csv(config.SUBMISSION_PATH, index=False)

print(f"\n  → Submission saved: {config.SUBMISSION_PATH}")
print(f"\n  Test Prediction Statistics:")
print(f"    Mean: {final_test.mean():.6f}")
print(f"    Std:  {final_test.std():.6f}")
print(f"    Min:  {final_test.min():.6f}")
print(f"    Max:  {final_test.max():.6f}")

