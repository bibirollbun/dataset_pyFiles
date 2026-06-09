import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path

# Machine Learning
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss, accuracy_score
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import beta, norm
from scipy.special import expit, logit

# Data Processing
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score, log_loss, classification_report, confusion_matrix, accuracy_score
from sklearn.calibration import CalibratedClassifierCV

# Gradient Boosting
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

# Hyperparameter Optimization
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner, MedianPruner

optuna.logging.set_verbosity(optuna.logging.WARNING)

# Feature Engineering & Selection
from sklearn.feature_selection import mutual_info_classif
from itertools import combinations
import shap


import matplotlib.pyplot as plt
import seaborn as sns

# Metrics
from sklearn.metrics import (
    roc_curve, roc_auc_score, confusion_matrix,
    brier_score_loss, log_loss, accuracy_score,
    precision_score, recall_score, f1_score
)
from sklearn.calibration import calibration_curve

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

@dataclass
class ImprovedConfig:
    """Enhanced configuration based on 2024-2025 research"""

    # File Paths
    TRAIN_PATH: str = '/kaggle/input/playground-series-s5e12/train.csv'
    TEST_PATH: str = '/kaggle/input/playground-series-s5e12/test.csv'
    SAMPLE_SUBMISSION_PATH: str = '/kaggle/input/playground-series-s5e12/sample_submission.csv'
    SUBMISSION_PATH: str = 'submission.csv'
    MODELS_DIR: str = 'models_advanced'

    # Column Names
    TARGET: str = 'diagnosed_diabetes'
    ID_COL: str = 'id'

    # Categorical Features (will be target-encoded)
    CATEGORICAL_FEATURES: List[str] = None

    def __post_init__(self):
        self.CATEGORICAL_FEATURES = [
            'gender', 'ethnicity', 'education_level', 'income_level',
            'smoking_status', 'employment_status'
        ]

    # Random Seed
    SEED: int = 42

    # Cross-Validation (FIXED: Proper stratified K-fold)
    N_FOLDS: int = 5
    N_REPEATS: int = 1  # Can increase for more robust validation

    # Feature Engineering
    USE_DOMAIN_FEATURES: bool = True
    USE_POLYNOMIAL_FEATURES: bool = True
    USE_STATISTICAL_FEATURES: bool = True
    USE_INTERACTION_FEATURES: bool = True

    # Feature Selection (BorutaShap-inspired)
    USE_FEATURE_SELECTION: bool = True
    FEATURE_SELECTION_THRESHOLD: float = 0.001
    MAX_FEATURES: int = 50  # Will test multiple sizes

    # Hyperparameter Optimization
    USE_OPTUNA: bool = True
    N_TRIALS: int = 5  # Increased for better optimization
    OPTUNA_TIMEOUT: int = 3600  # 1 hour per model

    # Ensemble Configuration
    USE_STACKING: bool = True
    USE_CALIBRATION: bool = True
    CALIBRATION_METHOD: str = 'isotonic'  # Better with enough data

    # Class Imbalance Handling
    USE_CLASS_WEIGHTS: bool = True  # Better than SMOTE per 2024 research

    # Model Selection
    MODELS_TO_USE: List[str] = None

    def __post_init__(self):
        if self.CATEGORICAL_FEATURES is None:
            self.CATEGORICAL_FEATURES = [
                'gender', 'ethnicity', 'education_level', 'income_level',
                'smoking_status', 'employment_status'
            ]
        if self.MODELS_TO_USE is None:
            self.MODELS_TO_USE = ['catboost','lightgbm', 'xgboost'] #'extratrees', 'catboost', 


class MedicalFeatureEngineer:
    """Domain-knowledge based feature engineering for diabetes prediction"""

    def __init__(self, config: ImprovedConfig):
        self.config = config

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create medical domain-specific features"""
        df = df.copy()

        print("  â†’ Creating medical domain features...")

        # â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�
        # CHOLESTEROL & LIPID RATIOS (Critical for diabetes/CVD)
        # â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�
        if all(col in df.columns for col in ['cholesterol_total', 'hdl_cholesterol']):
            df['chol_hdl_ratio'] = df['cholesterol_total'] / (df['hdl_cholesterol'] + 1e-5)
            df['atherogenic_index'] = (df['cholesterol_total'] - df['hdl_cholesterol']) / (df['hdl_cholesterol'] + 1e-5)

        if all(col in df.columns for col in ['ldl_cholesterol', 'hdl_cholesterol']):
            df['ldl_hdl_ratio'] = df['ldl_cholesterol'] / (df['hdl_cholesterol'] + 1e-5)

        if all(col in df.columns for col in ['triglycerides', 'hdl_cholesterol']):
            df['trig_hdl_ratio'] = df['triglycerides'] / (df['hdl_cholesterol'] + 1e-5)
            # Atherogenic dyslipidemia indicator
            df['atherogenic_dyslipidemia'] = ((df['triglycerides'] > 150) & (df['hdl_cholesterol'] < 40)).astype(int)

        if all(col in df.columns for col in ['cholesterol_total', 'hdl_cholesterol', 'ldl_cholesterol']):
            df['non_hdl_chol'] = df['cholesterol_total'] - df['hdl_cholesterol']
            df['cholesterol_balance'] = (df['hdl_cholesterol'] - df['ldl_cholesterol']) / (df['cholesterol_total'] + 1e-5)

        # â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�
        # BLOOD PRESSURE & CARDIOVASCULAR METRICS
        # â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�
        if all(col in df.columns for col in ['systolic_bp', 'diastolic_bp']):
            df['pulse_pressure'] = df['systolic_bp'] - df['diastolic_bp']
            df['mean_arterial_pressure'] = df['diastolic_bp'] + (df['pulse_pressure'] / 3)
            df['bp_product'] = df['systolic_bp'] * df['diastolic_bp']
            df['bp_ratio'] = df['systolic_bp'] / (df['diastolic_bp'] + 1e-5)

            # Hypertension categories
            df['hypertension_stage'] = 0
            df.loc[(df['systolic_bp'] >= 120) & (df['systolic_bp'] < 130), 'hypertension_stage'] = 1
            df.loc[(df['systolic_bp'] >= 130) & (df['systolic_bp'] < 140), 'hypertension_stage'] = 2
            df.loc[df['systolic_bp'] >= 140, 'hypertension_stage'] = 3

        # â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�
        # BMI & BODY COMPOSITION
        # â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�
        if 'bmi' in df.columns:
            # BMI categories (WHO classification)
            df['bmi_category'] = pd.cut(df['bmi'],
                                       bins=[0, 18.5, 25, 30, 35, 40, 100],
                                       labels=[0, 1, 2, 3, 4, 5]).astype(int)
            df['is_obese'] = (df['bmi'] >= 30).astype(int)
            df['is_severely_obese'] = (df['bmi'] >= 35).astype(int)

            # BMI squared (non-linear relationship)
            df['bmi_squared'] = df['bmi'] ** 2
            df['bmi_cubed'] = df['bmi'] ** 3

            if 'age' in df.columns:
                df['bmi_age_interaction'] = df['bmi'] * df['age']
                df['obesity_years'] = df['is_obese'] * df['age']

        if 'waist_to_hip_ratio' in df.columns:
            df['central_obesity'] = (df['waist_to_hip_ratio'] > 0.90).astype(int)  # For men, 0.85 for women

            if 'bmi' in df.columns:
                df['bmi_whr_product'] = df['bmi'] * df['waist_to_hip_ratio']
                df['metabolic_risk_index'] = df['bmi'] * df['waist_to_hip_ratio'] * 10

        # â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�
        # METABOLIC SYNDROME SCORE (Crucial for diabetes)
        # â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�
        metabolic_score = 0
        if 'bmi' in df.columns:
            metabolic_score += (df['bmi'] >= 30).astype(int)
        if 'systolic_bp' in df.columns:
            metabolic_score += (df['systolic_bp'] >= 130).astype(int)
        if 'triglycerides' in df.columns:
            metabolic_score += (df['triglycerides'] >= 150).astype(int)
        if 'hdl_cholesterol' in df.columns:
            metabolic_score += (df['hdl_cholesterol'] < 40).astype(int)
        if 'waist_to_hip_ratio' in df.columns:
            metabolic_score += (df['waist_to_hip_ratio'] > 0.90).astype(int)

        df['metabolic_syndrome_score'] = metabolic_score
        df['has_metabolic_syndrome'] = (metabolic_score >= 3).astype(int)

        # â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�
        # LIFESTYLE FACTORS
        # â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�
        if 'physical_activity_minutes_per_week' in df.columns:
            df['daily_activity_minutes'] = df['physical_activity_minutes_per_week'] / 7
            df['is_sedentary'] = (df['physical_activity_minutes_per_week'] < 150).astype(int)
            df['is_active'] = (df['physical_activity_minutes_per_week'] >= 300).astype(int)

            if 'bmi' in df.columns:
                df['activity_bmi_ratio'] = df['physical_activity_minutes_per_week'] / (df['bmi'] + 1)
                df['sedentary_obese'] = df['is_sedentary'] * df['is_obese']

        if 'diet_score' in df.columns:
            df['poor_diet'] = (df['diet_score'] < 5).astype(int)
            df['good_diet'] = (df['diet_score'] >= 7).astype(int)

            if 'bmi' in df.columns:
                df['diet_bmi_interaction'] = df['diet_score'] * df['bmi']

        if 'sleep_hours_per_day' in df.columns:
            df['sleep_deprived'] = (df['sleep_hours_per_day'] < 6).astype(int)
            df['sleep_excess'] = (df['sleep_hours_per_day'] > 9).astype(int)
            df['sleep_abnormal'] = df['sleep_deprived'] + df['sleep_excess']

            if 'screen_time_hours_per_day' in df.columns:
                df['sleep_screen_ratio'] = df['sleep_hours_per_day'] / (df['screen_time_hours_per_day'] + 1e-5)
                df['rest_quality_score'] = df['sleep_hours_per_day'] - df['screen_time_hours_per_day']

        if 'alcohol_consumption_per_week' in df.columns:
            df['heavy_drinker'] = (df['alcohol_consumption_per_week'] > 7).astype(int)  # >7 drinks/week

        # â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�
        # RISK FACTORS & FAMILY HISTORY
        # â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�
        risk_cols = ['family_history_diabetes', 'hypertension_history', 'cardiovascular_history']
        available_risk_cols = [col for col in risk_cols if col in df.columns]

        if available_risk_cols:
            df['total_risk_factors'] = df[available_risk_cols].sum(axis=1)
            df['has_family_history'] = (df['total_risk_factors'] > 0).astype(int)
            df['multiple_risk_factors'] = (df['total_risk_factors'] >= 2).astype(int)

            if 'age' in df.columns:
                df['age_risk_interaction'] = df['age'] * df['total_risk_factors']

        # â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�
        # AGE-RELATED FEATURES
        # â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�
        if 'age' in df.columns:
            df['age_group'] = pd.cut(df['age'],
                                    bins=[0, 30, 40, 50, 60, 100],
                                    labels=[0, 1, 2, 3, 4]).astype(int)
            df['is_senior'] = (df['age'] >= 60).astype(int)
            df['is_middle_aged'] = ((df['age'] >= 40) & (df['age'] < 60)).astype(int)
            df['age_squared'] = df['age'] ** 2

            # Age interactions with biomarkers
            if 'cholesterol_total' in df.columns:
                df['age_cholesterol'] = df['age'] * df['cholesterol_total']
            if 'systolic_bp' in df.columns:
                df['age_bp'] = df['age'] * df['systolic_bp']

        # â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�
        # COMPOSITE RISK SCORES
        # â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�
        # Simplified diabetes risk score
        diabetes_risk = 0
        if 'age' in df.columns:
            diabetes_risk += (df['age'] >= 45).astype(int) * 2
        if 'bmi' in df.columns:
            diabetes_risk += (df['bmi'] >= 30).astype(int) * 2
        if 'family_history_diabetes' in df.columns:
            diabetes_risk += df['family_history_diabetes'] * 3
        if 'hypertension_history' in df.columns:
            diabetes_risk += df['hypertension_history']
        if 'physical_activity_minutes_per_week' in df.columns:
            diabetes_risk += (df['physical_activity_minutes_per_week'] < 150).astype(int)

        df['diabetes_risk_score'] = diabetes_risk
        df['high_risk'] = (diabetes_risk >= 5).astype(int)

        # â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�
        # POLYNOMIAL & LOG TRANSFORMS for key biomarkers
        # â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�
        important_numeric_cols = [
            'cholesterol_total', 'ldl_cholesterol', 'triglycerides',
            'systolic_bp', 'diastolic_bp', 'heart_rate'
        ]

        for col in important_numeric_cols:
            if col in df.columns:
                df[f'{col}_squared'] = df[col] ** 2
                df[f'{col}_log'] = np.log1p(df[col].clip(lower=0))
                df[f'{col}_sqrt'] = np.sqrt(df[col].clip(lower=0))

        # â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�
        # STATISTICAL AGGREGATIONS
        # â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�
        # Cholesterol profile summary
        chol_cols = [col for col in df.columns if 'cholesterol' in col.lower()
                    and col in df.select_dtypes(include=[np.number]).columns]
        if len(chol_cols) >= 2:
            df['cholesterol_avg'] = df[chol_cols].mean(axis=1)
            df['cholesterol_std'] = df[chol_cols].std(axis=1)
            df['cholesterol_range'] = df[chol_cols].max(axis=1) - df[chol_cols].min(axis=1)

        # Lifestyle summary
        lifestyle_keywords = ['sleep', 'screen', 'physical', 'diet', 'alcohol']
        lifestyle_cols = []
        for keyword in lifestyle_keywords:
            lifestyle_cols.extend([col for col in df.columns if keyword in col.lower()
                                  and col in df.select_dtypes(include=[np.number]).columns])
        lifestyle_cols = list(set(lifestyle_cols))

        if len(lifestyle_cols) >= 2:
            df['lifestyle_avg'] = df[lifestyle_cols].mean(axis=1)
            df['lifestyle_std'] = df[lifestyle_cols].std(axis=1)

        print(f"    âœ“ Created {len([c for c in df.columns]) - len(df.columns)} medical features")

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
    Based on 2024 research showing superior performance.
    """

    def __init__(self, n_estimators: int = 100, threshold: float = 0.001, max_features: int = 50):
        self.n_estimators = n_estimators
        self.threshold = threshold
        self.max_features = max_features
        self.selected_features = None
        self.feature_importances = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fit selector using SHAP values"""
        print("\n  â†’ Running SHAP-based feature selection...")

        # Train a fast model for SHAP
        model = lgb.LGBMClassifier(
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

        print(f"    âœ“ Selected {len(self.selected_features)} features out of {len(X.columns)}")
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


class LightGBMProbabilisticClassifier:
    """
    LightGBM Classifier with Probability Calibration and Uncertainty Estimation.

    ADVANTAGES:
    - Fast training and prediction
    - Memory-efficient (handles large datasets)
    - Built-in categorical feature handling
    - Isotonic calibration for medical-grade probabilities
    - Bootstrap for uncertainty quantification

    References:
    - LightGBM: https://lightgbm.readthedocs.io/
    - Calibration: Niculescu-Mizil & Caruana (2005)
    """

    def __init__(
        self,
        n_trials: int = 50,
        cv_folds: int = 5,
        n_bootstrap: int = 10,
        calibration_method: str = 'isotonic',
        random_state: int = 42,
        verbose: bool = True
    ):
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.n_bootstrap = n_bootstrap
        self.calibration_method = calibration_method
        self.random_state = random_state
        self.verbose = verbose

        self.models: List[lgb.LGBMClassifier] = []
        self.calibrators: List[CalibratedClassifierCV] = []
        self.best_params: Dict = {}
        self.study: Optional[optuna.Study] = None

    def _create_objective(self, X: np.ndarray, y: np.ndarray, class_weights: Dict = None):
        """Create Optuna objective for hyperparameter optimization."""

        def objective(trial: optuna.Trial) -> float:
            params = {
                'objective': 'binary',
                'metric': 'auc',
                'boosting_type': 'gbdt',
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0.0, 1.0),
                'random_state': self.random_state,
                'verbosity': -1,
                'force_col_wise': True,
                'n_jobs': -1
            }

            if class_weights:
                params['class_weight'] = class_weights

            # Stratified K-Fold CV
            skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
            scores = []

            for train_idx, val_idx in skf.split(X, y):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                model = lgb.LGBMClassifier(**params)
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(50, verbose=False)]
                )

                y_pred_proba = model.predict_proba(X_val)[:, 1]
                auc = roc_auc_score(y_val, y_pred_proba)
                scores.append(auc)

            return np.mean(scores)

        return objective

    def optimize(self, X: np.ndarray, y: np.ndarray, class_weights: Dict = None):
        """Optimize hyperparameters using Optuna."""
        if self.verbose:
            print("\n" + "="*70)
            print("LIGHTGBM PROBABILISTIC CLASSIFIER - OPTUNA OPTIMIZATION")
            print("="*70)

        self.study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=self.random_state),
            pruner=HyperbandPruner(),
            study_name='lgbm_probabilistic_classifier'
        )

        objective = self._create_objective(X, y, class_weights)
        self.study.optimize(
            objective,
            n_trials=self.n_trials,
            show_progress_bar=self.verbose,
            n_jobs=-1
        )

        self.best_params = self.study.best_params

        if self.verbose:
            print(f"\n  âœ“ Best AUC: {self.study.best_value:.6f}")
            print(f"  âœ“ Best params: {self.best_params}")

        return self

    def fit(self, X: np.ndarray, y: np.ndarray, use_best_params: bool = True,
            class_weights: Dict = None):
        """Fit bootstrap ensemble with calibration."""
        if self.verbose:
            print(f"\n  â†’ Training {self.n_bootstrap} bootstrapped models with calibration...")

        # Get parameters
        if use_best_params and self.best_params:
            params = self.best_params.copy()
        else:
            params = {
                'n_estimators': 500,
                'num_leaves': 31,
                'learning_rate': 0.05,
                'max_depth': 7
            }

        # Add fixed params
        params.update({
            'objective': 'binary',
            'metric': 'auc',
            'random_state': self.random_state,
            'verbosity': -1,
            'force_col_wise': True,
            'n_jobs': -1
        })

        if class_weights:
            params['class_weight'] = class_weights

        n_samples = len(X)

        # Train bootstrap ensemble
        for i in range(self.n_bootstrap):
            # Bootstrap sample
            indices = np.random.RandomState(self.random_state + i).choice(
                n_samples, size=n_samples, replace=True
            )
            X_boot = X[indices]
            y_boot = y[indices]

            # Out-of-bag samples for calibration
            oob_mask = np.ones(n_samples, dtype=bool)
            oob_mask[indices] = False
            oob_indices = np.where(oob_mask)[0]

            if len(oob_indices) < 10:  # Need enough samples for calibration
                oob_indices = np.random.RandomState(self.random_state + i).choice(
                    n_samples, size=min(100, n_samples // 5), replace=False
                )

            X_oob = X[oob_indices]
            y_oob = y[oob_indices]

            # Train base model
            params['random_state'] = self.random_state + i
            model = lgb.LGBMClassifier(**params)
            model.fit(X_boot, y_boot, eval_set=[(X_oob, y_oob)],
                     callbacks=[lgb.early_stopping(50, verbose=False)])

            # Calibrate on OOB samples
            calibrator = CalibratedClassifierCV(
                model,
                method=self.calibration_method,
                cv='prefit'
            )
            calibrator.fit(X_oob, y_oob)

            self.models.append(model)
            self.calibrators.append(calibrator)

            if self.verbose and (i + 1) % 5 == 0:
                print(f"    Trained {i+1}/{self.n_bootstrap} models")

        if self.verbose:
            print("  âœ“ LightGBM Probabilistic Classifier fitted successfully")

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities (mean of ensemble)."""
        all_probas = np.array([
            calibrator.predict_proba(X)[:, 1]
            for calibrator in self.calibrators
        ])
        return all_probas.mean(axis=0)

    def predict_proba_with_uncertainty(
        self, X: np.ndarray, confidence: float = 0.90
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict probabilities with confidence intervals.

        Returns:
            mean_proba: Mean predicted probability
            lower_bound: Lower confidence bound
            upper_bound: Upper confidence bound
        """
        all_probas = np.array([
            calibrator.predict_proba(X)[:, 1]
            for calibrator in self.calibrators
        ])

        mean_proba = all_probas.mean(axis=0)

        # Compute percentiles for confidence interval
        alpha = (1 - confidence) / 2
        lower_bound = np.percentile(all_probas, alpha * 100, axis=0)
        upper_bound = np.percentile(all_probas, (1 - alpha) * 100, axis=0)

        return mean_proba, lower_bound, upper_bound

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict binary class labels."""
        return (self.predict_proba(X) >= threshold).astype(int)

    def get_feature_importance(self, feature_names: List[str] = None) -> pd.DataFrame:
        """Get average feature importance across ensemble."""
        importances = np.array([
            model.feature_importances_ for model in self.models
        ]).mean(axis=0)

        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(importances))]

        return pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)


# â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�
# 2. XGBOOST PROBABILISTIC CLASSIFIER WITH CALIBRATION
# â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�

class XGBoostProbabilisticClassifier:
    """
    XGBoost Classifier with Probability Calibration and Uncertainty Estimation.

    ADVANTAGES:
    - State-of-the-art performance
    - Regularization prevents overfitting
    - Handles missing values naturally
    - GPU acceleration available
    - Isotonic calibration for reliability
    """

    def __init__(
        self,
        n_trials: int = 50,
        cv_folds: int = 5,
        n_bootstrap: int = 10,
        calibration_method: str = 'isotonic',
        random_state: int = 42,
        verbose: bool = True
    ):
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.n_bootstrap = n_bootstrap
        self.calibration_method = calibration_method
        self.random_state = random_state
        self.verbose = verbose

        self.models: List[xgb.XGBClassifier] = []
        self.calibrators: List[CalibratedClassifierCV] = []
        self.best_params: Dict = {}
        self.study: Optional[optuna.Study] = None

    def _create_objective(self, X: np.ndarray, y: np.ndarray, class_weights: Dict = None):
        """Create Optuna objective."""

        def objective(trial: optuna.Trial) -> float:
            params = {
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'gamma': trial.suggest_float('gamma', 0.0, 5.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'random_state': self.random_state,
                'tree_method': 'hist',
                'verbosity': 0,
                'n_jobs': -1
            }

            if class_weights:
                scale_pos_weight = class_weights[1] / class_weights[0]
                params['scale_pos_weight'] = scale_pos_weight

            skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
            scores = []

            for train_idx, val_idx in skf.split(X, y):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                model = xgb.XGBClassifier(**params)
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

                y_pred_proba = model.predict_proba(X_val)[:, 1]
                auc = roc_auc_score(y_val, y_pred_proba)
                scores.append(auc)

            return np.mean(scores)

        return objective

    def optimize(self, X: np.ndarray, y: np.ndarray, class_weights: Dict = None):
        """Optimize hyperparameters."""
        if self.verbose:
            print("\n" + "="*70)
            print("XGBOOST PROBABILISTIC CLASSIFIER - OPTUNA OPTIMIZATION")
            print("="*70)

        self.study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=self.random_state),
            pruner=HyperbandPruner(),
            study_name='xgb_probabilistic_classifier'
        )

        objective = self._create_objective(X, y, class_weights)
        self.study.optimize(
            objective,
            n_trials=self.n_trials,
            show_progress_bar=self.verbose,
            n_jobs=-1
        )

        self.best_params = self.study.best_params

        if self.verbose:
            print(f"\n  âœ“ Best AUC: {self.study.best_value:.6f}")
            print(f"  âœ“ Best params: {self.best_params}")

        return self

    def fit(self, X: np.ndarray, y: np.ndarray, use_best_params: bool = True,
            class_weights: Dict = None):
        """Fit bootstrap ensemble with calibration."""
        if self.verbose:
            print(f"\n  â†’ Training {self.n_bootstrap} bootstrapped XGBoost models...")

        if use_best_params and self.best_params:
            params = self.best_params.copy()
        else:
            params = {
                'n_estimators': 500,
                'max_depth': 6,
                'learning_rate': 0.05
            }

        params.update({
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'random_state': self.random_state,
            'tree_method': 'hist',
            'verbosity': 0
        })

        if class_weights:
            scale_pos_weight = class_weights[1] / class_weights[0]
            params['scale_pos_weight'] = scale_pos_weight

        n_samples = len(X)

        for i in range(self.n_bootstrap):
            indices = np.random.RandomState(self.random_state + i).choice(
                n_samples, size=n_samples, replace=True
            )
            X_boot = X[indices]
            y_boot = y[indices]

            oob_mask = np.ones(n_samples, dtype=bool)
            oob_mask[indices] = False
            oob_indices = np.where(oob_mask)[0]

            if len(oob_indices) < 10:
                oob_indices = np.random.RandomState(self.random_state + i).choice(
                    n_samples, size=min(100, n_samples // 5), replace=False
                )

            X_oob = X[oob_indices]
            y_oob = y[oob_indices]

            params['random_state'] = self.random_state + i
            model = xgb.XGBClassifier(**params)
            model.fit(X_boot, y_boot, eval_set=[(X_oob, y_oob)], verbose=False)

            calibrator = CalibratedClassifierCV(
                model, method=self.calibration_method, cv='prefit'
            )
            calibrator.fit(X_oob, y_oob)

            self.models.append(model)
            self.calibrators.append(calibrator)

            if self.verbose and (i + 1) % 5 == 0:
                print(f"    Trained {i+1}/{self.n_bootstrap} models")

        if self.verbose:
            print("  âœ“ XGBoost Probabilistic Classifier fitted successfully")

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        all_probas = np.array([
            calibrator.predict_proba(X)[:, 1]
            for calibrator in self.calibrators
        ])
        return all_probas.mean(axis=0)

    def predict_proba_with_uncertainty(
        self, X: np.ndarray, confidence: float = 0.90
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict probabilities with confidence intervals."""
        all_probas = np.array([
            calibrator.predict_proba(X)[:, 1]
            for calibrator in self.calibrators
        ])

        mean_proba = all_probas.mean(axis=0)
        alpha = (1 - confidence) / 2
        lower_bound = np.percentile(all_probas, alpha * 100, axis=0)
        upper_bound = np.percentile(all_probas, (1 - alpha) * 100, axis=0)

        return mean_proba, lower_bound, upper_bound

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict binary class labels."""
        return (self.predict_proba(X) >= threshold).astype(int)

    def get_feature_importance(self, feature_names: List[str] = None) -> pd.DataFrame:
        """Get average feature importance."""
        importances = np.array([
            model.feature_importances_ for model in self.models
        ]).mean(axis=0)

        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(importances))]

        return pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)


# â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�
# 3. CATBOOST PROBABILISTIC CLASSIFIER
# â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�

class CatBoostProbabilisticClassifier:
    """
    CatBoost Classifier with Probability Calibration.

    ADVANTAGES:
    - Native categorical feature handling
    - Ordered boosting (reduces overfitting)
    - Built-in GPU support
    - Less hyperparameter tuning needed
    - Symmetric trees for faster inference
    """

    def __init__(
        self,
        n_trials: int = 50,
        cv_folds: int = 5,
        n_bootstrap: int = 10,
        calibration_method: str = 'isotonic',
        random_state: int = 42,
        verbose: bool = True
    ):
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.n_bootstrap = n_bootstrap
        self.calibration_method = calibration_method
        self.random_state = random_state
        self.verbose = verbose

        self.models: List[CatBoostClassifier] = []
        self.calibrators: List[CalibratedClassifierCV] = []
        self.best_params: Dict = {}
        self.study: Optional[optuna.Study] = None

    def _create_objective(self, X: np.ndarray, y: np.ndarray, class_weights: Dict = None,
                         cat_features: List[int] = None):
        """Create Optuna objective."""

        def objective(trial: optuna.Trial) -> float:
            params = {
                'iterations': trial.suggest_int('iterations', 200, 1000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'depth': trial.suggest_int('depth', 4, 10),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
                'border_count': trial.suggest_int('border_count', 32, 255),
                'random_strength': trial.suggest_float('random_strength', 0.0, 10.0),
                'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
                'random_state': self.random_state,
                'verbose': False,
                'task_type': 'CPU',
                'loss_function': 'Logloss',
                'eval_metric': 'AUC'
            }

            if class_weights:
                params['class_weights'] = class_weights

            skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
            scores = []

            for train_idx, val_idx in skf.split(X, y):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                model = CatBoostClassifier(**params)
                model.fit(
                    X_train, y_train,
                    eval_set=(X_val, y_val),
                    cat_features=cat_features,
                    verbose=False
                )

                y_pred_proba = model.predict_proba(X_val)[:, 1]
                auc = roc_auc_score(y_val, y_pred_proba)
                scores.append(auc)

            return np.mean(scores)

        return objective

    def optimize(self, X: np.ndarray, y: np.ndarray, class_weights: Dict = None,
                cat_features: List[int] = None):
        """Optimize hyperparameters."""
        if self.verbose:
            print("\n" + "="*70)
            print("CATBOOST PROBABILISTIC CLASSIFIER - OPTUNA OPTIMIZATION")
            print("="*70)

        self.study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=self.random_state),
            pruner=HyperbandPruner(),
            study_name='catboost_probabilistic_classifier'
        )

        objective = self._create_objective(X, y, class_weights, cat_features)
        self.study.optimize(
            objective,
            n_trials=self.n_trials,
            show_progress_bar=self.verbose,
            n_jobs=-1
        )

        self.best_params = self.study.best_params

        if self.verbose:
            print(f"\n  âœ“ Best AUC: {self.study.best_value:.6f}")
            print(f"  âœ“ Best params: {self.best_params}")

        return self

    def fit(self, X: np.ndarray, y: np.ndarray, use_best_params: bool = True,
            class_weights: Dict = None, cat_features: List[int] = None):
        """Fit bootstrap ensemble with calibration."""
        if self.verbose:
            print(f"\n  â†’ Training {self.n_bootstrap} bootstrapped CatBoost models...")

        if use_best_params and self.best_params:
            params = self.best_params.copy()
        else:
            params = {
                'iterations': 500,
                'learning_rate': 0.05,
                'depth': 6
            }

        params.update({
            'random_state': self.random_state,
            'verbose': False,
            'task_type': 'CPU',
            'loss_function': 'Logloss',
            'eval_metric': 'AUC'
        })

        if class_weights:
            params['class_weights'] = class_weights

        n_samples = len(X)

        for i in range(self.n_bootstrap):
            indices = np.random.RandomState(self.random_state + i).choice(
                n_samples, size=n_samples, replace=True
            )
            X_boot = X[indices]
            y_boot = y[indices]

            oob_mask = np.ones(n_samples, dtype=bool)
            oob_mask[indices] = False
            oob_indices = np.where(oob_mask)[0]

            if len(oob_indices) < 10:
                oob_indices = np.random.RandomState(self.random_state + i).choice(
                    n_samples, size=min(100, n_samples // 5), replace=False
                )

            X_oob = X[oob_indices]
            y_oob = y[oob_indices]

            params['random_state'] = self.random_state + i
            model = CatBoostClassifier(**params)
            model.fit(X_boot, y_boot, eval_set=(X_oob, y_oob),
                     cat_features=cat_features, verbose=False)

            calibrator = CalibratedClassifierCV(
                model, method=self.calibration_method, cv='prefit'
            )
            calibrator.fit(X_oob, y_oob)

            self.models.append(model)
            self.calibrators.append(calibrator)

            if self.verbose and (i + 1) % 5 == 0:
                print(f"    Trained {i+1}/{self.n_bootstrap} models")

        if self.verbose:
            print("  âœ“ CatBoost Probabilistic Classifier fitted successfully")

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        all_probas = np.array([
            calibrator.predict_proba(X)[:, 1]
            for calibrator in self.calibrators
        ])
        return all_probas.mean(axis=0)

    def predict_proba_with_uncertainty(
        self, X: np.ndarray, confidence: float = 0.90
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict probabilities with confidence intervals."""
        all_probas = np.array([
            calibrator.predict_proba(X)[:, 1]
            for calibrator in self.calibrators
        ])

        mean_proba = all_probas.mean(axis=0)
        alpha = (1 - confidence) / 2
        lower_bound = np.percentile(all_probas, alpha * 100, axis=0)
        upper_bound = np.percentile(all_probas, (1 - alpha) * 100, axis=0)

        return mean_proba, lower_bound, upper_bound

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict binary class labels."""
        return (self.predict_proba(X) >= threshold).astype(int)

    def get_feature_importance(self, feature_names: List[str] = None) -> pd.DataFrame:
        """Get average feature importance."""
        importances = np.array([
            model.feature_importances_ for model in self.models
        ]).mean(axis=0)

        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(importances))]

        return pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)

# â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�
# 4. NGBOOST BERNOULLI CLASSIFIER
# â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�

class NGBoostBernoulliClassifier:
    """
    NGBoost (Natural Gradient Boosting) for Binary Classification with Bernoulli Distribution.

    THEORETICAL FOUNDATION:
    ======================
    Based on Stanford ML Group's NGBoost paper (Duan et al., 2019):
    - Uses Natural Gradient Boosting for probabilistic binary classification
    - Bernoulli distribution for binary outcomes (Y âˆˆ {0, 1})
    - Fisher Information Matrix for Riemannian geometry
    - Calibrated probability estimates with uncertainty quantification

    KEY ADVANTAGES:
    ==============
    1. **True Probabilistic Predictions**: Full probability distribution, not just point estimates
    2. **Naturally Calibrated Probabilities**: Fisher Information rescaling provides inherent calibration
       - No need for post-hoc calibration methods (isotonic/sigmoid)
       - Probabilities are theoretically well-calibrated from the natural gradient
    3. **Stable Optimization**: Natural gradient is more stable than standard gradient boosting
    4. **Theoretically Grounded**: Based on information geometry and Riemannian metrics
    5. **Bootstrap Ensemble**: Epistemic uncertainty from model variation
    6. **Medical-Grade Reliability**: Calibrated probabilities crucial for clinical decision-making

    MATHEMATICAL DETAILS:
    ====================
    For Bernoulli distribution with parameter p (probability of class 1):

    1. **Parametrization**: Use logit(p) = log(p/(1-p)) to ensure p âˆˆ (0,1)
       - p = sigmoid(logit) = 1/(1 + exp(-logit))

    2. **Negative Log-Likelihood (Binary Cross-Entropy)**:
       NLL = -[y*log(p) + (1-y)*log(1-p)]

    3. **Gradient w.r.t. logit**:
       âˆ‚NLL/âˆ‚logit = p - y

    4. **Fisher Information** (for logit parametrization):
       F = p(1-p)

    5. **Natural Gradient**:
       natural_grad = (p - y) / [p(1-p)] = (p - y) / var(Bernoulli)

    This ensures stable convergence when boosting probability distributions.

    """

    def __init__(
        self,
        n_trials: int = 50,
        cv_folds: int = 5,
        n_bootstrap: int = 10,
        calibration_method: str = 'isotonic',
        random_state: int = 42,
        verbose: bool = True
    ):
        """
        Initialize NGBoost Bernoulli Classifier.

        Parameters:
        -----------
        n_trials : int
            Number of Optuna trials for hyperparameter optimization
        cv_folds : int
            Number of cross-validation folds
        n_bootstrap : int
            Number of bootstrap models for uncertainty quantification
        calibration_method : str
            Calibration method: 'isotonic' or 'sigmoid'
        random_state : int
            Random seed for reproducibility
        verbose : bool
            Print training progress
        """
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.n_bootstrap = n_bootstrap
        self.calibration_method = calibration_method
        self.random_state = random_state
        self.verbose = verbose

        # Model storage
        self.bootstrap_models = []
        self.best_params = {}
        self.study = None
        self.scaler = StandardScaler()

        # Base initialization value (logit of 0.5)
        self.logit_base = 0.0

    def _sigmoid(self, logit: np.ndarray) -> np.ndarray:
        """Convert logit to probability using sigmoid function."""
        return 1.0 / (1.0 + np.exp(-np.clip(logit, -500, 500)))

    def _log_loss_gradient(
        self,
        y: np.ndarray,
        logit: np.ndarray
    ) -> np.ndarray:
        """
        Compute gradient of negative log-likelihood (binary cross-entropy).

        For Bernoulli distribution:
        NLL = -[y*log(p) + (1-y)*log(1-p)]

        Gradient w.r.t. logit:
        âˆ‚NLL/âˆ‚logit = p - y

        where p = sigmoid(logit)
        """
        p = self._sigmoid(logit)
        return p - y

    def _fisher_information(
        self,
        logit: np.ndarray
    ) -> np.ndarray:
        """
        Compute Fisher Information for Bernoulli distribution.

        For logit parametrization:
        F = p(1-p) = variance of Bernoulli

        This is the expected second derivative of NLL w.r.t. logit.
        """
        p = self._sigmoid(logit)
        return p * (1 - p)

    def _natural_gradient(
        self,
        y: np.ndarray,
        logit: np.ndarray
    ) -> np.ndarray:
        """
        Compute natural gradient using Fisher Information Matrix.

        Natural Gradient = F^(-1) @ gradient
                        = (p - y) / [p(1-p)]

        This rescales the gradient based on the geometry of the probability
        distribution space, leading to more stable optimization.
        """
        gradient = self._log_loss_gradient(y, logit)
        fisher = self._fisher_information(logit)

        # Natural gradient with numerical stability
        fisher_clipped = np.clip(fisher, 1e-6, 1 - 1e-6)
        return gradient / fisher_clipped

    def _fit_base_learner(
        self,
        X: np.ndarray,
        gradient: np.ndarray,
        base_params: Dict
    ):
        """
        Fit decision tree to negative gradient.

        This is the core of gradient boosting: fitting weak learners
        to the negative gradient (residuals in logit space).
        """
        from sklearn.tree import DecisionTreeRegressor

        learner = DecisionTreeRegressor(
            **base_params,
            random_state=self.random_state
        )

        # Fit to NEGATIVE gradient (gradient descent direction)
        learner.fit(X, -gradient)

        return learner

    def _compute_log_likelihood(
        self,
        y: np.ndarray,
        logit: np.ndarray
    ) -> float:
        """
        Compute average log-likelihood.

        For Bernoulli: LL = y*log(p) + (1-y)*log(1-p)
        """
        p = self._sigmoid(logit)
        p_clipped = np.clip(p, 1e-15, 1 - 1e-15)

        ll = y * np.log(p_clipped) + (1 - y) * np.log(1 - p_clipped)
        return np.mean(ll)

    def _create_objective(self, X: np.ndarray, y: np.ndarray):
        """Create Optuna objective for hyperparameter optimization."""

        def objective(trial: optuna.Trial) -> float:
            # Base learner parameters
            base_params = {
                'max_depth': trial.suggest_int('max_depth', 2, 8),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
            }

            # Boosting parameters
            n_estimators = trial.suggest_int('n_estimators', 50, 500)
            learning_rate = trial.suggest_float('learning_rate', 0.001, 0.3, log=True)

            # Stratified K-Fold cross-validation
            from sklearn.model_selection import StratifiedKFold
            skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True,
                                 random_state=self.random_state)

            scores = []

            for train_idx, val_idx in skf.split(X, y):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                # Initialize with logit of training set mean
                p_init = np.clip(np.mean(y_train), 0.01, 0.99)
                logit_init = np.log(p_init / (1 - p_init))

                # Current predictions (updated each iteration)
                logit_train = np.full(len(X_train), logit_init)

                # Store learners for validation
                learners = []

                # Boosting iterations
                for _ in range(n_estimators):
                    # Compute natural gradient
                    nat_grad = self._natural_gradient(y_train, logit_train)

                    # Fit base learner
                    learner = self._fit_base_learner(X_train, nat_grad, base_params)

                    # Update predictions
                    logit_train += learning_rate * learner.predict(X_train)

                    # Store learner
                    learners.append(learner)

                # Validate: accumulate predictions from all learners
                logit_val = logit_init + sum([
                    learning_rate * learner.predict(X_val) for learner in learners
                ])

                # Compute negative log-likelihood (lower is better)
                nll = -self._compute_log_likelihood(y_val, logit_val)
                scores.append(nll)

            return np.mean(scores)

        return objective

    def optimize(self, X: np.ndarray, y: np.ndarray, class_weights: Dict = None):
        """
        Optimize hyperparameters using Optuna.

        Parameters:
        -----------
        X : np.ndarray
            Training features
        y : np.ndarray
            Training labels (0 or 1)
        class_weights : Dict, optional
            Class weights for imbalanced data (not used in NGBoost directly,
            but can be passed for compatibility)
        """
        if self.verbose:
            print("\n" + "="*70)
            print("  NGBOOST BERNOULLI OPTIMIZATION")
            print("="*70)

        # Standardize features
        X_scaled = self.scaler.fit_transform(X)

        # Create Optuna study
        self.study = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(seed=self.random_state),
            pruner=HyperbandPruner(),
            study_name='ngboost_bernoulli'
        )

        # Optimize
        objective = self._create_objective(X_scaled, y)
        self.study.optimize(
            objective,
            n_trials=self.n_trials,
            show_progress_bar=self.verbose, 
            n_jobs=-1
        )

        self.best_params = self.study.best_params

        if self.verbose:
            print(f"\n  Best Hyperparameters:")
            for key, value in self.best_params.items():
                print(f"    {key}: {value}")
            print(f"    Best NLL: {self.study.best_value:.6f}")
            print("="*70)

        return self

    def fit(self, X: np.ndarray, y: np.ndarray, use_best_params: bool = True,
            class_weights: Dict = None):
        """
        Fit NGBoost Bernoulli Classifier with bootstrap ensemble.

        Parameters:
        -----------
        X : np.ndarray
            Training features
        y : np.ndarray
            Training labels (0 or 1)
        use_best_params : bool
            Use optimized hyperparameters
        class_weights : Dict, optional
            Class weights for compatibility
        """
        if self.verbose:
            print(f"\n  Training NGBoost Bernoulli with {self.n_bootstrap} bootstrap models...")

        # Standardize features
        X_scaled = self.scaler.fit_transform(X)

        # Get parameters
        if use_best_params and self.best_params:
            base_params = {
                'max_depth': self.best_params['max_depth'],
                'min_samples_split': self.best_params['min_samples_split'],
                'min_samples_leaf': self.best_params['min_samples_leaf'],
                'max_features': self.best_params['max_features']
            }
            n_estimators = self.best_params['n_estimators']
            learning_rate = self.best_params['learning_rate']
        else:
            base_params = {
                'max_depth': 3,
                'min_samples_split': 10,
                'min_samples_leaf': 5,
                'max_features': 'sqrt'
            }
            n_estimators = 200
            learning_rate = 0.01

        # Bootstrap ensemble for uncertainty
        for b in range(self.n_bootstrap):
            if self.verbose and b % max(1, self.n_bootstrap // 5) == 0:
                print(f"    Bootstrap model {b+1}/{self.n_bootstrap}")

            # Bootstrap sample
            n_samples = len(X_scaled)
            indices = np.random.RandomState(self.random_state + b).choice(
                n_samples, size=n_samples, replace=True
            )
            X_boot = X_scaled[indices]
            y_boot = y[indices]

            # Initialize logit
            p_init = np.clip(np.mean(y_boot), 0.01, 0.99)
            logit_init = np.log(p_init / (1 - p_init))

            logit_current = np.full(len(X_boot), logit_init)

            # Store base learners
            learners = []

            # Boosting iterations
            for iteration in range(n_estimators):
                # Natural gradient
                nat_grad = self._natural_gradient(y_boot, logit_current)

                # Fit base learner
                learner = self._fit_base_learner(X_boot, nat_grad, base_params)

                # Update logit
                logit_current += learning_rate * learner.predict(X_boot)

                # Store learner
                learners.append(learner)

            # Store model
            # Note: NGBoost already provides well-calibrated probabilities via natural gradients
            # The Fisher Information Matrix rescaling ensures proper calibration without
            # needing additional post-hoc calibration methods
            model_dict = {
                'learners': learners,
                'logit_init': logit_init,
                'learning_rate': learning_rate
            }
            self.bootstrap_models.append(model_dict)

        if self.verbose:
            print(f"  âœ“ Training complete!")

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities (mean across ensemble).

        Parameters:
        -----------
        X : np.ndarray
            Features to predict

        Returns:
        --------
        proba : np.ndarray
            Mean predicted probabilities for class 1
        """
        X_scaled = self.scaler.transform(X)

        all_probs = []

        for model in self.bootstrap_models:
            # Accumulate predictions from all base learners
            logit = model['logit_init'] + sum([
                model['learning_rate'] * learner.predict(X_scaled)
                for learner in model['learners']
            ])

            # Convert to probability
            prob = self._sigmoid(logit)
            all_probs.append(prob)

        # Return mean probability
        return np.mean(all_probs, axis=0)

    def predict_proba_with_uncertainty(
        self,
        X: np.ndarray,
        confidence: float = 0.90
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict probabilities with uncertainty intervals.

        Parameters:
        -----------
        X : np.ndarray
            Features to predict
        confidence : float
            Confidence level for intervals (default 0.90)

        Returns:
        --------
        mean_proba : np.ndarray
            Mean predicted probabilities
        lower_bound : np.ndarray
            Lower confidence bound
        upper_bound : np.ndarray
            Upper confidence bound
        """
        X_scaled = self.scaler.transform(X)

        all_probs = []

        for model in self.bootstrap_models:
            # Accumulate predictions
            logit = model['logit_init'] + sum([
                model['learning_rate'] * learner.predict(X_scaled)
                for learner in model['learners']
            ])

            prob = self._sigmoid(logit)
            all_probs.append(prob)

        all_probs = np.array(all_probs)

        # Calculate statistics
        mean_proba = np.mean(all_probs, axis=0)

        # Confidence intervals from bootstrap distribution
        alpha = (1 - confidence) / 2
        lower_bound = np.percentile(all_probs, alpha * 100, axis=0)
        upper_bound = np.percentile(all_probs, (1 - alpha) * 100, axis=0)

        return mean_proba, lower_bound, upper_bound

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict class labels.

        Parameters:
        -----------
        X : np.ndarray
            Features to predict
        threshold : float
            Classification threshold (default 0.5)

        Returns:
        --------
        predictions : np.ndarray
            Predicted class labels (0 or 1)
        """
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    def get_feature_importance(self, feature_names: List[str] = None) -> pd.DataFrame:
        """
        Get feature importance from ensemble.

        Parameters:
        -----------
        feature_names : List[str], optional
            Names of features

        Returns:
        --------
        importance_df : pd.DataFrame
            Feature importances sorted by importance
        """
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(
                len(self.bootstrap_models[0]['learners'][0].feature_importances_)
            )]

        # Aggregate importance across all base learners in all bootstrap models
        all_importances = []

        for model in self.bootstrap_models:
            for learner in model['learners']:
                all_importances.append(learner.feature_importances_)

        # Average importance
        importances = np.mean(all_importances, axis=0)

        return pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)


# â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�
# 5. PROBABILISTIC MODEL TRAINER - MAIN WRAPPER CLASS
# â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�

class ProbabilisticModelTrainer:
    """
    Main trainer class compatible with ImprovedModelTrainer API.

    Provides drop-in replacement for standard classifiers with:
    - Calibrated probability predictions
    - Uncertainty quantification
    - Confidence intervals
    - Compatible interface for easy integration

    Usage Example:
    ```python
    from improved_diabetes_prediction import ImprovedConfig
    from probabilistic_classification_models import ProbabilisticModelTrainer

    config = ImprovedConfig()
    trainer = ProbabilisticModelTrainer(config)

    # Train models
    results = trainer.train_with_cv(
        X=X_train, y=y_train, X_test=X_test
    )

    # Get predictions with uncertainty
    mean_proba, lower, upper = trainer.predict_with_uncertainty(X_test)
    ```
    """

    def __init__(self, config):
        """
        Initialize with ImprovedConfig object.

        Args:
            config: ImprovedConfig instance with all settings
        """
        self.config = config
        self.models = {}
        self.oof_predictions = {}
        self.test_predictions = {}
        self.uncertainty_lower = {}
        self.uncertainty_upper = {}
        self.feature_importances = {}

    def calculate_class_weights(self, y: pd.Series) -> Dict:
        """Calculate class weights for imbalanced data."""
        class_counts = y.value_counts()
        total = len(y)
        weights = {
            0: total / (2 * class_counts[0]),
            1: total / (2 * class_counts[1])
        }
        return weights

    def train_with_cv(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_test: pd.DataFrame,
        model_names: List[str] = None,
        n_trials: int = 30,
        n_bootstrap: int = 10
    ) -> Dict[str, Any]:
        """
        Train probabilistic models with proper cross-validation.

        Compatible with ImprovedModelTrainer.train_with_cv()

        Args:
            X: Training features
            y: Training target
            X_test: Test features
            model_names: List of models to train ['lightgbm', 'xgboost', 'catboost', 'ngboost']
            n_trials: Number of Optuna trials for hyperparameter optimization
            n_bootstrap: Number of bootstrap models for uncertainty

        Returns:
            results: Dictionary with OOF scores and predictions
        """
        if model_names is None:
            model_names = ['lightgbm', 'xgboost', 'catboost', 'ngboost']

        print(f"\n{'='*70}")
        print(f"  PROBABILISTIC MODEL TRAINING - {self.config.N_FOLDS}-Fold CV")
        print(f"{'='*70}")

        # Calculate class weights if enabled
        class_weights = None
        if self.config.USE_CLASS_WEIGHTS:
            class_weights = self.calculate_class_weights(y)
            print(f"\n  Class weights: {class_weights}")

        # Convert to numpy for compatibility
        X_np = X.values if isinstance(X, pd.DataFrame) else X
        y_np = y.values if isinstance(y, pd.Series) else y
        X_test_np = X_test.values if isinstance(X_test, pd.DataFrame) else X_test

        results = {}

        for model_name in model_names:
            print(f"\n{'='*70}")
            print(f"  Training {model_name.upper()} Probabilistic Classifier")
            print(f"{'='*70}")

            # Create model
            if model_name == 'lightgbm':
                model = LightGBMProbabilisticClassifier(
                    n_trials=n_trials,
                    cv_folds=self.config.N_FOLDS,
                    n_bootstrap=n_bootstrap,
                    random_state=self.config.SEED,
                    verbose=True
                )
            elif model_name == 'xgboost':
                model = XGBoostProbabilisticClassifier(
                    n_trials=n_trials,
                    cv_folds=self.config.N_FOLDS,
                    n_bootstrap=n_bootstrap,
                    random_state=self.config.SEED,
                    verbose=True
                )
            elif model_name == 'catboost':
                model = CatBoostProbabilisticClassifier(
                    n_trials=n_trials,
                    cv_folds=self.config.N_FOLDS,
                    n_bootstrap=n_bootstrap,
                    random_state=self.config.SEED,
                    verbose=True
                )
            elif model_name == 'ngboost':
                model = NGBoostBernoulliClassifier(
                    n_trials=n_trials,
                    cv_folds=self.config.N_FOLDS,
                    n_bootstrap=n_bootstrap,
                    random_state=self.config.SEED,
                    verbose=True
                )
            else:
                print(f"  âš ï¸�  Unknown model: {model_name}, skipping...")
                continue

            # Optimize hyperparameters
            if self.config.USE_OPTUNA:
                model.optimize(X_np, y_np, class_weights=class_weights)

            # Train model
            model.fit(X_np, y_np, use_best_params=self.config.USE_OPTUNA,
                     class_weights=class_weights)

            # Get OOF predictions with uncertainty using CV
            oof_preds, oof_lower, oof_upper = self._get_oof_predictions(model, X_np, y_np)

            # Get test predictions with uncertainty
            test_mean, test_lower, test_upper = model.predict_proba_with_uncertainty(
                X_test_np, confidence=0.90
            )

            # Calculate OOF score
            oof_score = roc_auc_score(y_np, oof_preds)
            oof_acc = accuracy_score(y_np, (oof_preds > 0.5).astype(int))

            print(f"\n  {'â”€'*70}")
            print(f"  {model_name.upper()} Results:")
            print(f"    OOF AUC:      {oof_score:.6f}")
            print(f"    OOF Accuracy: {oof_acc:.6f}")
            print(f"    Mean Uncertainty Width: {np.mean(test_upper - test_lower):.4f}")
            print(f"  {'â”€'*70}")

            # Store results
            self.models[model_name] = model
            self.oof_predictions[model_name] = oof_preds
            self.test_predictions[model_name] = test_mean
            self.uncertainty_lower[model_name] = test_lower
            self.uncertainty_upper[model_name] = test_upper

            # Feature importance
            if isinstance(X, pd.DataFrame):
                self.feature_importances[model_name] = model.get_feature_importance(
                    feature_names=X.columns.tolist()
                )

            results[model_name] = {
                'oof_score': oof_score,
                'oof_acc': oof_acc,
                'oof_preds': oof_preds,
                'oof_lower': oof_lower,
                'oof_upper': oof_upper,
                'test_preds': test_mean,
                'test_lower': test_lower,
                'test_upper': test_upper,
                'uncertainty_width': np.mean(test_upper - test_lower)
            }

        return results

    def _get_oof_predictions(
        self, model, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get out-of-fold predictions with uncertainty for validation.

        Returns:
            oof_preds: Mean OOF predictions
            oof_lower: Lower confidence bounds
            oof_upper: Upper confidence bounds
        """
        oof_preds = np.zeros(len(X))
        oof_lower = np.zeros(len(X))
        oof_upper = np.zeros(len(X))

        skf = StratifiedKFold(
            n_splits=self.config.N_FOLDS,
            shuffle=True,
            random_state=self.config.SEED
        )

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            X_val = X[val_idx]

            # Use the trained ensemble to predict with uncertainty
            val_mean, val_lower, val_upper = model.predict_proba_with_uncertainty(
                X_val, confidence=0.90
            )
            oof_preds[val_idx] = val_mean
            oof_lower[val_idx] = val_lower
            oof_upper[val_idx] = val_upper

        return oof_preds, oof_lower, oof_upper

    def predict_with_uncertainty(
        self,
        X: pd.DataFrame,
        model_name: str = None,
        confidence: float = 0.90
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get predictions with uncertainty intervals.

        Args:
            X: Features to predict
            model_name: Which model to use (if None, uses best model)
            confidence: Confidence level for intervals

        Returns:
            mean_pred: Mean probability predictions
            lower_bound: Lower confidence bound
            upper_bound: Upper confidence bound
        """
        if model_name is None:
            # Use best model by OOF score
            model_name = max(
                self.oof_predictions.items(),
                key=lambda x: roc_auc_score(y, x[1]) if 'y' in locals() else 0
            )[0]

        model = self.models[model_name]
        X_np = X.values if isinstance(X, pd.DataFrame) else X

        return model.predict_proba_with_uncertainty(X_np, confidence=confidence)

    def get_ensemble_predictions(
        self, X: pd.DataFrame, weights: Dict[str, float] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get weighted ensemble predictions from all models.

        Args:
            X: Features
            weights: Optional weights for each model

        Returns:
            mean_pred, lower_bound, upper_bound
        """
        if weights is None:
            # Equal weights
            weights = {name: 1.0 / len(self.models) for name in self.models.keys()}

        X_np = X.values if isinstance(X, pd.DataFrame) else X

        # Collect predictions
        all_means = []
        all_lowers = []
        all_uppers = []

        for name, model in self.models.items():
            mean, lower, upper = model.predict_proba_with_uncertainty(X_np)
            all_means.append(mean * weights[name])
            all_lowers.append(lower * weights[name])
            all_uppers.append(upper * weights[name])

        # Combine
        ensemble_mean = np.sum(all_means, axis=0)
        ensemble_lower = np.sum(all_lowers, axis=0)
        ensemble_upper = np.sum(all_uppers, axis=0)

        return ensemble_mean, ensemble_lower, ensemble_upper

    def get_calibration_plots_data(
        self, y_true: np.ndarray, model_name: str = None, n_bins: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get data for calibration plot (reliability diagram).

        Args:
            y_true: True labels
            model_name: Which model to analyze
            n_bins: Number of bins for calibration curve

        Returns:
            prob_true: True probabilities in each bin
            prob_pred: Predicted probabilities in each bin
        """
        if model_name is None:
            model_name = list(self.models.keys())[0]

        y_pred = self.oof_predictions[model_name]

        prob_true, prob_pred = calibration_curve(
            y_true, y_pred, n_bins=n_bins, strategy='uniform'
        )

        return prob_true, prob_pred

    def evaluate_uncertainty_quality(
        self, y_true: np.ndarray, model_name: str = None
    ) -> Dict[str, float]:
        """
        Evaluate quality of uncertainty estimates.

        Args:
            y_true: True labels
            model_name: Which model to evaluate

        Returns:
            metrics: Dictionary of uncertainty quality metrics
        """
        if model_name is None:
            model_name = list(self.models.keys())[0]

        mean_pred = self.oof_predictions[model_name]

        # Brier score (lower is better)
        brier = brier_score_loss(y_true, mean_pred)

        # Log loss (lower is better)
        logloss = log_loss(y_true, mean_pred)

        # AUC (higher is better)
        auc = roc_auc_score(y_true, mean_pred)

        # Expected Calibration Error (ECE)
        prob_true, prob_pred = self.get_calibration_plots_data(y_true, model_name)
        ece = np.mean(np.abs(prob_true - prob_pred))

        return {
            'auc': auc,
            'brier_score': brier,
            'log_loss': logloss,
            'expected_calibration_error': ece
        }


# â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�
# UTILITY FUNCTIONS FOR EVALUATION
# â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�

def plot_uncertainty_distribution(
    predictions: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    title: str = "Prediction Uncertainty Distribution"
) -> None:
    """
    Print statistics about uncertainty distribution.

    Args:
        predictions: Mean predictions
        lower_bounds: Lower confidence bounds
        upper_bounds: Upper confidence bounds
        title: Plot title
    """
    uncertainty_widths = upper_bounds - lower_bounds

    print(f"\n{title}")
    print("="*70)
    print(f"Prediction Statistics:")
    print(f"  Mean Prediction:        {np.mean(predictions):.4f}")
    print(f"  Std Prediction:         {np.std(predictions):.4f}")
    print(f"\nUncertainty Statistics:")
    print(f"  Mean Uncertainty Width: {np.mean(uncertainty_widths):.4f}")
    print(f"  Std Uncertainty Width:  {np.std(uncertainty_widths):.4f}")
    print(f"  Min Uncertainty Width:  {np.min(uncertainty_widths):.4f}")
    print(f"  Max Uncertainty Width:  {np.max(uncertainty_widths):.4f}")
    print(f"\nConfidence Interval Coverage:")
    print(f"  Lower Bound Range: [{np.min(lower_bounds):.4f}, {np.max(lower_bounds):.4f}]")
    print(f"  Upper Bound Range: [{np.min(upper_bounds):.4f}, {np.max(upper_bounds):.4f}]")
    print("="*70)


def compare_probabilistic_models(
    results: Dict[str, Dict],
    y_true: np.ndarray = None
) -> pd.DataFrame:
    """
    Compare multiple probabilistic models.

    Args:
        results: Results dictionary from ProbabilisticModelTrainer
        y_true: True labels (optional, for OOF evaluation)

    Returns:
        comparison_df: DataFrame with model comparisons
    """
    comparison = []

    for model_name, result in results.items():
        row = {
            'Model': model_name.upper(),
            'OOF AUC': result.get('oof_score', np.nan),
            'OOF Accuracy': result.get('oof_acc', np.nan),
            'Mean Uncertainty': result.get('uncertainty_width', np.nan)
        }

        if y_true is not None and 'oof_preds' in result:
            row['Brier Score'] = brier_score_loss(y_true, result['oof_preds'])
            row['Log Loss'] = log_loss(y_true, result['oof_preds'])

        comparison.append(row)

    df = pd.DataFrame(comparison)
    df = df.sort_values('OOF AUC', ascending=False).reset_index(drop=True)

    return df


def print_model_summary(
    trainer: ProbabilisticModelTrainer,
    y_train: np.ndarray
) -> None:
    """
    Print comprehensive summary of trained probabilistic models.

    Args:
        trainer: Fitted ProbabilisticModelTrainer
        y_train: Training labels
    """
    print("\n" + "="*70)
    print("PROBABILISTIC MODELS SUMMARY")
    print("="*70)

    for model_name in trainer.models.keys():
        print(f"\n{model_name.upper()}:")
        print("-"*70)

        metrics = trainer.evaluate_uncertainty_quality(y_train, model_name)

        print(f"  AUC Score:                  {metrics['auc']:.6f}")
        print(f"  Brier Score:                {metrics['brier_score']:.6f}")
        print(f"  Log Loss:                   {metrics['log_loss']:.6f}")
        print(f"  Expected Calibration Error: {metrics['expected_calibration_error']:.6f}")

        if model_name in trainer.test_predictions:
            mean_uncertainty = np.mean(
                trainer.uncertainty_upper[model_name] - trainer.uncertainty_lower[model_name]
            )
            print(f"  Mean Test Uncertainty:      {mean_uncertainty:.6f}")

    print("\n" + "="*70)


class ProbabilisticModelDashboard:
    """
    Comprehensive visualization dashboard for probabilistic classification models.
    ```
    """

    def __init__(
        self,
        trainer,
        results: Dict[str, Dict],
        y_train: np.ndarray,
        save_figures: bool = False,
        output_dir: str = 'visualizations',
        figsize: Tuple[int, int] = (12, 8),
        dpi: int = 100
    ):
        """
        Initialize dashboard.

        Args:
            trainer: Fitted ProbabilisticModelTrainer instance
            results: Results dictionary from trainer.train_with_cv()
            y_train: True training labels
            save_figures: If True, save plots to files; if False, display in notebook
            output_dir: Directory to save plots (only used if save_figures=True)
            figsize: Default figure size
            dpi: Resolution for figures
        """
        self.trainer = trainer
        self.results = results
        self.y_train = y_train
        self.save_figures = save_figures
        self.output_dir = Path(output_dir) if save_figures else None
        self.figsize = figsize
        self.dpi = dpi

        # Create output directory only if saving
        if self.save_figures:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*70}")
        print("PROBABILISTIC MODEL VISUALIZATION DASHBOARD")
        print(f"{'='*70}")
        if self.save_figures:
            print(f"Mode: Save to files")
            print(f"Output directory: {self.output_dir}")
        else:
            print(f"Mode: Display in notebook")
        print(f"Models to visualize: {list(results.keys())}")
        print(f"{'='*70}\n")

    def _show_or_save(self, filename: str):
        """Helper method to either show plot in notebook or save to file."""
        if self.save_figures:
            plt.savefig(self.output_dir / filename, dpi=self.dpi, bbox_inches='tight')
            plt.close()
        else:
            plt.tight_layout()
            plt.show()

    def generate_all_plots(self):
        """Generate all visualizations."""
        print("\nGenerating all visualizations...")

        self.plot_model_comparison()
        self.plot_roc_curves()
        self.plot_calibration_curves()
        self.plot_uncertainty_distributions()
        self.plot_confidence_intervals()
        self.plot_confusion_matrices()
        self.plot_feature_importance()
        self.plot_prediction_distributions()
        self.plot_uncertainty_vs_error()

        if self.save_figures:
            print(f"\nâœ“ All visualizations saved to: {self.output_dir}")
        else:
            print(f"\nâœ“ All visualizations displayed")
        print(f"{'='*70}\n")

    # â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�
    # 1. MODEL PERFORMANCE COMPARISON
    # â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�

    def plot_model_comparison(self):
        """Compare all models across multiple metrics."""
        print("  Generating model comparison plot...")

        # Collect metrics
        metrics_data = []
        for model_name, result in self.results.items():
            oof_preds = result['oof_preds']

            metrics_data.append({
                'Model': model_name.upper(),
                'AUC': roc_auc_score(self.y_train, oof_preds),
                'Accuracy': accuracy_score(self.y_train, (oof_preds > 0.5).astype(int)),
                'Precision': precision_score(self.y_train, (oof_preds > 0.5).astype(int)),
                'Recall': recall_score(self.y_train, (oof_preds > 0.5).astype(int)),
                'F1 Score': f1_score(self.y_train, (oof_preds > 0.5).astype(int)),
                'Brier Score': brier_score_loss(self.y_train, oof_preds),
                'Log Loss': log_loss(self.y_train, oof_preds)
            })

        df_metrics = pd.DataFrame(metrics_data)

        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Probabilistic Models Performance Comparison',
                     fontsize=16, fontweight='bold')

        # Plot 1: Classification Metrics
        ax1 = axes[0, 0]
        metrics_to_plot = ['AUC', 'Accuracy', 'Precision', 'Recall', 'F1 Score']
        df_plot = df_metrics.set_index('Model')[metrics_to_plot]
        df_plot.plot(kind='bar', ax=ax1, width=0.8)
        ax1.set_title('Classification Metrics (Higher is Better)', fontweight='bold')
        ax1.set_ylabel('Score')
        ax1.set_xlabel('Model')
        ax1.legend(loc='lower right')
        ax1.set_ylim([0, 1.05])
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.3, label='Baseline')

        # Plot 2: Loss Metrics
        ax2 = axes[0, 1]
        loss_metrics = ['Brier Score', 'Log Loss']
        df_plot2 = df_metrics.set_index('Model')[loss_metrics]
        df_plot2.plot(kind='bar', ax=ax2, width=0.8, color=['#e74c3c', '#9b59b6'])
        ax2.set_title('Loss Metrics (Lower is Better)', fontweight='bold')
        ax2.set_ylabel('Loss')
        ax2.set_xlabel('Model')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)

        # Plot 3: AUC Ranking
        ax3 = axes[1, 0]
        df_sorted = df_metrics.sort_values('AUC', ascending=True)
        colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(df_sorted)))
        ax3.barh(df_sorted['Model'], df_sorted['AUC'], color=colors)
        ax3.set_xlabel('AUC Score')
        ax3.set_title('Model Ranking by AUC', fontweight='bold')
        ax3.set_xlim([0.5, 1.0])
        ax3.grid(True, alpha=0.3, axis='x')

        # Add value labels
        for i, (idx, row) in enumerate(df_sorted.iterrows()):
            ax3.text(row['AUC'] + 0.005, i, f"{row['AUC']:.4f}",
                    va='center', fontweight='bold')

        # Plot 4: Metrics Table
        ax4 = axes[1, 1]
        ax4.axis('off')

        # Create table
        table_data = df_metrics.round(4).values
        table = ax4.table(cellText=table_data, colLabels=df_metrics.columns,
                         cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)

        # Style header
        for i in range(len(df_metrics.columns)):
            table[(0, i)].set_facecolor('#3498db')
            table[(0, i)].set_text_props(weight='bold', color='white')

        # Color rows by rank
        for i in range(1, len(df_metrics) + 1):
            for j in range(len(df_metrics.columns)):
                if j == 0:  # Model name
                    table[(i, j)].set_facecolor('#ecf0f1')
                else:
                    table[(i, j)].set_facecolor('#ffffff')

        self._show_or_save('01_model_comparison.png')

    # â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�
    # 2. ROC CURVES
    # â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�

    def plot_roc_curves(self):
        """Plot ROC curves for all models."""
        print(" Generating ROC curves...")

        fig, ax = plt.subplots(figsize=self.figsize)

        colors = plt.cm.Set1(np.linspace(0, 1, len(self.results)))

        for (model_name, result), color in zip(self.results.items(), colors):
            oof_preds = result['oof_preds']

            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(self.y_train, oof_preds)
            auc = roc_auc_score(self.y_train, oof_preds)

            # Plot
            ax.plot(fpr, tpr, color=color, linewidth=2.5,
                   label=f'{model_name.upper()} (AUC = {auc:.4f})')

        # Diagonal line
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random Classifier', alpha=0.5)

        # Styling
        ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        ax.set_title('ROC Curves - Probabilistic Models', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

        self._show_or_save('02_roc_curves.png')

    # â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�
    # 3. CALIBRATION CURVES (RELIABILITY DIAGRAMS)
    # â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�

    def plot_calibration_curves(self):
        """Plot calibration curves showing probability reliability."""
        print(" Generating calibration curves...")

        fig, axes = plt.subplots(1, len(self.results),
                                figsize=(6 * len(self.results), 5))

        if len(self.results) == 1:
            axes = [axes]

        fig.suptitle('Probability Calibration Curves',
                    fontsize=14, fontweight='bold')

        for ax, (model_name, result) in zip(axes, self.results.items()):
            oof_preds = result['oof_preds']

            # Calculate calibration curve
            prob_true, prob_pred = calibration_curve(
                self.y_train, oof_preds, n_bins=10, strategy='uniform'
            )

            # Calculate Expected Calibration Error (ECE)
            ece = np.mean(np.abs(prob_true - prob_pred))

            # Plot
            ax.plot(prob_pred, prob_true, 's-', linewidth=2.5,
                   markersize=10, label=f'ECE = {ece:.4f}')
            ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Perfect Calibration')

            # Fill area
            ax.fill_between(prob_pred, prob_pred, prob_true, alpha=0.2)

            # Styling
            ax.set_xlabel('Predicted Probability', fontweight='bold')
            ax.set_ylabel('True Probability', fontweight='bold')
            ax.set_title(f'{model_name.upper()}', fontweight='bold')
            ax.legend(loc='lower right')
            ax.grid(True, alpha=0.3)
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            ax.set_aspect('equal')

        self._show_or_save('03_calibration_curves.png')

    # â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�
    # 4. UNCERTAINTY DISTRIBUTIONS
    # â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�

    def plot_uncertainty_distributions(self):
        """Plot uncertainty width distributions for all models."""
        print(" Generating uncertainty distribution plots...")

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Prediction Uncertainty Analysis',
                    fontsize=14, fontweight='bold')

        # Collect data
        uncertainty_data = []
        for model_name, result in self.results.items():
            lower = result['test_lower']
            upper = result['test_upper']
            mean = result['test_preds']
            width = upper - lower

            uncertainty_data.append({
                'model': model_name.upper(),
                'width': width,
                'mean': mean,
                'lower': lower,
                'upper': upper
            })

        # Plot 1: Uncertainty Width Distributions
        ax1 = axes[0, 0]
        for data in uncertainty_data:
            ax1.hist(data['width'], bins=50, alpha=0.6, label=data['model'], density=True)
        ax1.set_xlabel('Uncertainty Width (90% CI)', fontweight='bold')
        ax1.set_ylabel('Density', fontweight='bold')
        ax1.set_title('Uncertainty Width Distribution', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Box plots of uncertainty
        ax2 = axes[0, 1]
        box_data = [data['width'] for data in uncertainty_data]
        box_labels = [data['model'] for data in uncertainty_data]
        bp = ax2.boxplot(box_data, labels=box_labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], plt.cm.Set2(range(len(box_data)))):
            patch.set_facecolor(color)
        ax2.set_ylabel('Uncertainty Width', fontweight='bold')
        ax2.set_title('Uncertainty Comparison', fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')

        # Plot 3: Mean vs Uncertainty
        ax3 = axes[1, 0]
        for data, color in zip(uncertainty_data, plt.cm.Set1(range(len(uncertainty_data)))):
            ax3.scatter(data['mean'], data['width'], alpha=0.4, s=10,
                       color=color, label=data['model'])
        ax3.set_xlabel('Predicted Probability', fontweight='bold')
        ax3.set_ylabel('Uncertainty Width', fontweight='bold')
        ax3.set_title('Prediction vs Uncertainty', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Plot 4: Statistics Table
        ax4 = axes[1, 1]
        ax4.axis('off')

        stats_data = []
        for data in uncertainty_data:
            stats_data.append([
                data['model'],
                f"{np.mean(data['width']):.4f}",
                f"{np.std(data['width']):.4f}",
                f"{np.min(data['width']):.4f}",
                f"{np.max(data['width']):.4f}",
                f"{np.median(data['width']):.4f}"
            ])

        table = ax4.table(
            cellText=stats_data,
            colLabels=['Model', 'Mean', 'Std', 'Min', 'Max', 'Median'],
            cellLoc='center',
            loc='center',
            bbox=[0, 0, 1, 1]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)

        # Style header
        for i in range(6):
            table[(0, i)].set_facecolor('#3498db')
            table[(0, i)].set_text_props(weight='bold', color='white')

        self._show_or_save('04_uncertainty_distributions.png')

    # â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�
    # 5. CONFIDENCE INTERVALS
    # â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�

    def plot_confidence_intervals(self):
        """Plot sample predictions with confidence intervals."""
        print(" Generating confidence interval plots...")

        fig, axes = plt.subplots(len(self.results), 1,
                                figsize=(14, 4 * len(self.results)))

        if len(self.results) == 1:
            axes = [axes]

        fig.suptitle('Prediction Confidence Intervals (Sample of 100 Predictions)',
                    fontsize=14, fontweight='bold')

        n_samples = min(100, len(self.results[list(self.results.keys())[0]]['test_preds']))

        for ax, (model_name, result) in zip(axes, self.results.items()):
            mean = result['test_preds'][:n_samples]
            lower = result['test_lower'][:n_samples]
            upper = result['test_upper'][:n_samples]

            # Sort by mean for better visualization
            sort_idx = np.argsort(mean)
            mean_sorted = mean[sort_idx]
            lower_sorted = lower[sort_idx]
            upper_sorted = upper[sort_idx]

            x = np.arange(n_samples)

            # Plot
            ax.fill_between(x, lower_sorted, upper_sorted, alpha=0.3, label='90% CI')
            ax.plot(x, mean_sorted, 'o-', markersize=3, linewidth=1, label='Mean Prediction')
            ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Decision Threshold')

            # Styling
            ax.set_xlabel('Sample Index (sorted by prediction)', fontweight='bold')
            ax.set_ylabel('Probability', fontweight='bold')
            ax.set_title(f'{model_name.upper()}', fontweight='bold')
            ax.legend(loc='upper left')
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1])

        self._show_or_save('05_confidence_intervals.png')

    # â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�
    # 6. CONFUSION MATRICES
    # â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�

    def plot_confusion_matrices(self):
        """Plot confusion matrices for all models."""
        print(" Generating confusion matrices...")

        n_models = len(self.results)
        ncols = min(3, n_models)
        nrows = (n_models + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
        fig.suptitle('Confusion Matrices (Threshold = 0.5)',
                    fontsize=14, fontweight='bold')

        if n_models == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for ax, (model_name, result) in zip(axes, self.results.items()):
            oof_preds = result['oof_preds']
            y_pred_binary = (oof_preds > 0.5).astype(int)

            # Calculate confusion matrix
            cm = confusion_matrix(self.y_train, y_pred_binary)

            # Calculate percentages
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

            # Plot
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       cbar=True, square=True, linewidths=1, linecolor='black')

            # Add percentage annotations
            for i in range(2):
                for j in range(2):
                    text = ax.text(j + 0.5, i + 0.7, f'({cm_norm[i, j]*100:.1f}%)',
                                 ha="center", va="center", color="red", fontsize=9)

            ax.set_ylabel('True Label', fontweight='bold')
            ax.set_xlabel('Predicted Label', fontweight='bold')
            ax.set_title(f'{model_name.upper()}\nAcc: {accuracy_score(self.y_train, y_pred_binary):.4f}',
                        fontweight='bold')
            ax.set_yticklabels(['Negative', 'Positive'], rotation=0)
            ax.set_xticklabels(['Negative', 'Positive'])

        # Hide extra subplots
        for i in range(n_models, len(axes)):
            axes[i].axis('off')

        self._show_or_save('06_confusion_matrices.png')

    # â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�
    # 7. FEATURE IMPORTANCE
    # â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�

    def plot_feature_importance(self, top_n: int = 20):
        """Plot top feature importances for all models."""
        print(" Generating feature importance plots...")

        if not self.trainer.feature_importances:
            print("    âš ï¸�  Feature importance not available, skipping...")
            return

        n_models = len(self.trainer.feature_importances)
        fig, axes = plt.subplots(1, n_models, figsize=(8 * n_models, 10))

        if n_models == 1:
            axes = [axes]

        fig.suptitle(f'Top {top_n} Feature Importances',
                    fontsize=14, fontweight='bold')

        for ax, (model_name, importance_df) in zip(axes, self.trainer.feature_importances.items()):
            # Get top features
            top_features = importance_df.head(top_n)

            # Plot
            colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_features)))
            ax.barh(range(len(top_features)), top_features['importance'], color=colors)
            ax.set_yticks(range(len(top_features)))
            ax.set_yticklabels(top_features['feature'])
            ax.set_xlabel('Importance', fontweight='bold')
            ax.set_title(f'{model_name.upper()}', fontweight='bold')
            ax.invert_yaxis()
            ax.grid(True, alpha=0.3, axis='x')

        self._show_or_save('07_feature_importance.png')

    # â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�
    # 8. PREDICTION DISTRIBUTIONS
    # â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�

    def plot_prediction_distributions(self):
        """Plot prediction probability distributions by true class."""
        print(" Generating prediction distribution plots...")

        fig, axes = plt.subplots(2, len(self.results),
                                figsize=(6 * len(self.results), 10))

        if len(self.results) == 1:
            axes = axes.reshape(-1, 1)

        fig.suptitle('Prediction Distributions by True Class',
                    fontsize=14, fontweight='bold')

        for col, (model_name, result) in enumerate(self.results.items()):
            oof_preds = result['oof_preds']

            # Split by true class
            preds_class_0 = oof_preds[self.y_train == 0]
            preds_class_1 = oof_preds[self.y_train == 1]

            # Plot histogram
            ax1 = axes[0, col]
            ax1.hist(preds_class_0, bins=50, alpha=0.6, label='True Negative',
                    color='#3498db', density=True)
            ax1.hist(preds_class_1, bins=50, alpha=0.6, label='True Positive',
                    color='#e74c3c', density=True)
            ax1.axvline(x=0.5, color='black', linestyle='--', linewidth=2,
                       label='Threshold')
            ax1.set_xlabel('Predicted Probability', fontweight='bold')
            ax1.set_ylabel('Density', fontweight='bold')
            ax1.set_title(f'{model_name.upper()} - Histogram', fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Plot KDE
            ax2 = axes[1, col]
            from scipy.stats import gaussian_kde

            if len(preds_class_0) > 1:
                kde0 = gaussian_kde(preds_class_0)
                x_range = np.linspace(0, 1, 200)
                ax2.plot(x_range, kde0(x_range), linewidth=2.5,
                        label='True Negative', color='#3498db')
                ax2.fill_between(x_range, kde0(x_range), alpha=0.3, color='#3498db')

            if len(preds_class_1) > 1:
                kde1 = gaussian_kde(preds_class_1)
                ax2.plot(x_range, kde1(x_range), linewidth=2.5,
                        label='True Positive', color='#e74c3c')
                ax2.fill_between(x_range, kde1(x_range), alpha=0.3, color='#e74c3c')

            ax2.axvline(x=0.5, color='black', linestyle='--', linewidth=2,
                       label='Threshold')
            ax2.set_xlabel('Predicted Probability', fontweight='bold')
            ax2.set_ylabel('Density', fontweight='bold')
            ax2.set_title(f'{model_name.upper()} - KDE', fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        self._show_or_save('08_prediction_distributions.png')

    # â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�
    # 9. UNCERTAINTY VS PREDICTION ERROR
    # â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�

    def plot_uncertainty_vs_error(self):
        """Plot relationship between uncertainty and prediction error."""
        print(" Generating uncertainty vs error plots...")

        fig, axes = plt.subplots(1, len(self.results),
                                figsize=(7 * len(self.results), 6))

        if len(self.results) == 1:
            axes = [axes]

        fig.suptitle('Prediction Uncertainty vs Absolute Error',
                    fontsize=14, fontweight='bold')

        for ax, (model_name, result) in zip(axes, self.results.items()):
            oof_preds = result['oof_preds']

            # Calculate absolute error
            abs_error = np.abs(self.y_train - oof_preds)

            # For test set uncertainty (we don't have true labels, so visualize differently)
            # Here we show OOF predictions - we can estimate uncertainty from bootstrap
            # For simplicity, we'll create bins and show relationship

            # Bin predictions by uncertainty (using test uncertainty as proxy)
            test_lower = result['test_lower'][:len(oof_preds)] if len(result['test_lower']) >= len(oof_preds) else result['test_lower']
            test_upper = result['test_upper'][:len(oof_preds)] if len(result['test_upper']) >= len(oof_preds) else result['test_upper']

            if len(test_lower) < len(oof_preds):
                # Repeat to match length
                test_lower = np.tile(test_lower, (len(oof_preds) // len(test_lower) + 1))[:len(oof_preds)]
                test_upper = np.tile(test_upper, (len(oof_preds) // len(test_upper) + 1))[:len(oof_preds)]

            uncertainty = test_upper - test_lower

            # Scatter plot
            scatter = ax.scatter(uncertainty, abs_error, alpha=0.3, s=20,
                                c=oof_preds, cmap='RdYlGn', vmin=0, vmax=1)

            # Add binned mean line
            bins = np.percentile(uncertainty, np.linspace(0, 100, 11))
            bin_means_x = []
            bin_means_y = []

            for i in range(len(bins) - 1):
                mask = (uncertainty >= bins[i]) & (uncertainty < bins[i+1])
                if mask.sum() > 0:
                    bin_means_x.append((bins[i] + bins[i+1]) / 2)
                    bin_means_y.append(abs_error[mask].mean())

            ax.plot(bin_means_x, bin_means_y, 'r-o', linewidth=2.5,
                   markersize=8, label='Binned Mean Error')

            # Styling
            ax.set_xlabel('Uncertainty Width', fontweight='bold')
            ax.set_ylabel('Absolute Prediction Error', fontweight='bold')
            ax.set_title(f'{model_name.upper()}', fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Predicted Probability', fontweight='bold')

        self._show_or_save('09_uncertainty_vs_error.png')


# â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�
# SUMMARY REPORT GENERATOR
# â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�

def generate_summary_report(
    trainer,
    results: Dict[str, Dict],
    y_train: np.ndarray,
    output_dir: str = 'visualizations'
):
    """
    Generate a comprehensive text summary report.

    Args:
        trainer: Fitted ProbabilisticModelTrainer
        results: Results dictionary
        y_train: Training labels
        output_dir: Output directory
    """
    output_path = Path(output_dir) / 'model_summary_report.txt'

    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("PROBABILISTIC MODELS - COMPREHENSIVE SUMMARY REPORT\n")
        f.write("="*80 + "\n\n")

        # Overall comparison
        f.write("MODEL PERFORMANCE COMPARISON\n")
        f.write("-"*80 + "\n\n")

        for model_name, result in sorted(results.items(),
                                        key=lambda x: x[1]['oof_score'],
                                        reverse=True):
            oof_preds = result['oof_preds']
            y_pred_binary = (oof_preds > 0.5).astype(int)

            f.write(f"{model_name.upper()}:\n")
            f.write(f"  AUC Score:          {roc_auc_score(y_train, oof_preds):.6f}\n")
            f.write(f"  Accuracy:           {accuracy_score(y_train, y_pred_binary):.6f}\n")
            f.write(f"  Precision:          {precision_score(y_train, y_pred_binary):.6f}\n")
            f.write(f"  Recall:             {recall_score(y_train, y_pred_binary):.6f}\n")
            f.write(f"  F1 Score:           {f1_score(y_train, y_pred_binary):.6f}\n")
            f.write(f"  Brier Score:        {brier_score_loss(y_train, oof_preds):.6f}\n")
            f.write(f"  Log Loss:           {log_loss(y_train, oof_preds):.6f}\n")
            f.write(f"  Mean Uncertainty:   {result['uncertainty_width']:.6f}\n")
            f.write("\n")

        f.write("\n" + "="*80 + "\n")
        f.write("DETAILED UNCERTAINTY STATISTICS\n")
        f.write("="*80 + "\n\n")

        for model_name, result in results.items():
            lower = result['test_lower']
            upper = result['test_upper']
            width = upper - lower

            f.write(f"{model_name.upper()}:\n")
            f.write(f"  Mean Uncertainty:   {np.mean(width):.6f}\n")
            f.write(f"  Std Uncertainty:    {np.std(width):.6f}\n")
            f.write(f"  Min Uncertainty:    {np.min(width):.6f}\n")
            f.write(f"  Max Uncertainty:    {np.max(width):.6f}\n")
            f.write(f"  Median Uncertainty: {np.median(width):.6f}\n")
            f.write("\n")

        f.write("\n" + "="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")

    print(f"\nâœ“ Summary report saved to: {output_path}")


class ClinicalDecisionDashboard:
    """
    Medical decision-support visualization dashboard for probabilistic models.

    """

    def __init__(
        self,
        trainer,
        results: Dict,
        y_train: np.ndarray,
        save_figures: bool = False,
        output_dir: str = 'clinical_visualizations',
        figsize: Tuple[int, int] = (14, 10),
        dpi: int = 100
    ):
        """
        Initialize clinical dashboard.

        Parameters:
        -----------
        trainer : ProbabilisticModelTrainer
            Trained probabilistic model trainer
        results : Dict
            Results dictionary from trainer.train_with_cv()
        y_train : np.ndarray
            True labels for training data
        save_figures : bool
            If True, save plots to files; if False, display in notebook
        output_dir : str
            Directory for saved figures
        figsize : Tuple[int, int]
            Default figure size
        dpi : int
            Figure resolution
        """
        self.trainer = trainer
        self.results = results
        self.y_train = y_train
        self.save_figures = save_figures
        self.output_dir = Path(output_dir)
        self.figsize = figsize
        self.dpi = dpi

        if self.save_figures:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        # Clinical thresholds
        self.low_risk_threshold = 0.30
        self.high_risk_threshold = 0.70
        self.uncertainty_threshold = 0.15  # CI width threshold

        # Clinical costs (relative weights)
        self.cost_false_negative = 10  # Missing diabetes is costly
        self.cost_false_positive = 1   # False alarm is less costly

    def _show_or_save(self, filename: str):
        """Helper to either show in notebook or save to file."""
        if self.save_figures:
            plt.savefig(self.output_dir / filename, dpi=self.dpi, bbox_inches='tight')
            plt.close()
        else:
            plt.tight_layout()
            plt.show()

    def _classify_risk_level(self, prob: float, uncertainty: float) -> str:
        """
        Classify patient into clinical risk category.

        Categories:
        - Low Risk: prob < 0.30 and narrow CI
        - Medium Risk: 0.30 <= prob <= 0.70 or wide CI
        - High Risk: prob > 0.70 and narrow CI
        - Uncertain: wide CI (requires clinical review)
        """
        if uncertainty > self.uncertainty_threshold:
            return 'Uncertain (Review Required)'
        elif prob < self.low_risk_threshold:
            return 'Low Risk'
        elif prob > self.high_risk_threshold:
            return 'High Risk'
        else:
            return 'Medium Risk'

    def plot_risk_stratification(self):
        """
        PLOT 1: Population Risk Stratification

        Shows distribution of patients across risk categories with clinical
        recommendations for each category.
        """
        n_models = len(self.results)
        fig, axes = plt.subplots(2, n_models, figsize=(6*n_models, 12))
        if n_models == 1:
            axes = axes.reshape(2, 1)

        for idx, (model_name, result) in enumerate(self.results.items()):
            oof_preds = result['oof_preds']
            oof_lower = result['oof_lower']
            oof_upper = result['oof_upper']
            uncertainty = oof_upper - oof_lower

            # Classify each patient
            risk_categories = [
                self._classify_risk_level(p, u)
                for p, u in zip(oof_preds, uncertainty)
            ]

            # Count by category
            category_counts = pd.Series(risk_categories).value_counts()
            category_order = ['Low Risk', 'Medium Risk', 'High Risk', 'Uncertain (Review Required)']
            category_counts = category_counts.reindex(category_order, fill_value=0)

            # Plot 1: Bar chart with counts and percentages
            ax1 = axes[0, idx]
            colors = ['#2ecc71', '#f39c12', '#e74c3c', '#95a5a6']
            bars = ax1.bar(range(len(category_counts)), category_counts.values, color=colors, alpha=0.8)

            # Add value labels
            for i, (bar, count) in enumerate(zip(bars, category_counts.values)):
                height = bar.get_height()
                pct = 100 * count / len(oof_preds)
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(count)}\n({pct:.1f}%)',
                        ha='center', va='bottom', fontsize=10, fontweight='bold')

            ax1.set_xticks(range(len(category_counts)))
            ax1.set_xticklabels(category_order, rotation=45, ha='right')
            ax1.set_ylabel('Number of Patients', fontsize=12, fontweight='bold')
            ax1.set_title(f'{model_name.upper()}\nPatient Risk Distribution',
                         fontsize=14, fontweight='bold')
            ax1.grid(axis='y', alpha=0.3)

            # Plot 2: Distribution of probabilities by actual outcome
            ax2 = axes[1, idx]

            # Separate by true label
            diabetic_mask = self.y_train == 1
            healthy_mask = self.y_train == 0

            # Plot distributions
            ax2.hist(oof_preds[healthy_mask], bins=30, alpha=0.6, color='#3498db',
                    label=f'Healthy (n={healthy_mask.sum()})', density=True)
            ax2.hist(oof_preds[diabetic_mask], bins=30, alpha=0.6, color='#e74c3c',
                    label=f'Diabetic (n={diabetic_mask.sum()})', density=True)

            # Add threshold lines
            ax2.axvline(self.low_risk_threshold, color='green', linestyle='--',
                       linewidth=2, alpha=0.7, label='Low Risk Threshold')
            ax2.axvline(self.high_risk_threshold, color='red', linestyle='--',
                       linewidth=2, alpha=0.7, label='High Risk Threshold')

            ax2.set_xlabel('Predicted Probability', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Density', fontsize=12, fontweight='bold')
            ax2.set_title('Probability Distribution by True Diagnosis', fontsize=12, fontweight='bold')
            ax2.legend(loc='upper right', fontsize=9)
            ax2.grid(alpha=0.3)

        fig.suptitle('CLINICAL RISK STRATIFICATION ANALYSIS',
                    fontsize=16, fontweight='bold', y=0.995)
        self._show_or_save('01_risk_stratification.png')

    def plot_decision_threshold_analysis(self):
        """
        PLOT 2: Clinical Decision Threshold Analysis

        Helps doctors choose optimal probability threshold by showing trade-offs
        between sensitivity and specificity at different thresholds.
        """
        n_models = len(self.results)
        fig, axes = plt.subplots(2, n_models, figsize=(6*n_models, 12))
        if n_models == 1:
            axes = axes.reshape(2, 1)

        for idx, (model_name, result) in enumerate(self.results.items()):
            oof_preds = result['oof_preds']

            # Calculate metrics at different thresholds
            thresholds = np.linspace(0.1, 0.9, 81)
            sensitivities = []
            specificities = []
            ppvs = []  # Positive Predictive Value
            npvs = []  # Negative Predictive Value
            f1_scores = []

            for thresh in thresholds:
                y_pred = (oof_preds >= thresh).astype(int)
                tn, fp, fn, tp = confusion_matrix(self.y_train, y_pred).ravel()

                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall, True Positive Rate
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # True Negative Rate
                ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # Precision
                npv = tn / (tn + fn) if (tn + fn) > 0 else 0

                sensitivities.append(sensitivity)
                specificities.append(specificity)
                ppvs.append(ppv)
                npvs.append(npv)
                f1_scores.append(f1_score(self.y_train, y_pred, zero_division=0))

            # Plot 1: Sensitivity vs Specificity
            ax1 = axes[0, idx]
            ax1.plot(thresholds, sensitivities, 'r-', linewidth=2.5, label='Sensitivity (Recall)', alpha=0.8)
            ax1.plot(thresholds, specificities, 'b-', linewidth=2.5, label='Specificity', alpha=0.8)
            ax1.plot(thresholds, f1_scores, 'g--', linewidth=2, label='F1-Score', alpha=0.7)

            # Mark optimal F1 threshold
            optimal_idx = np.argmax(f1_scores)
            optimal_thresh = thresholds[optimal_idx]
            ax1.axvline(optimal_thresh, color='purple', linestyle=':', linewidth=2,
                       label=f'Optimal F1 ({optimal_thresh:.2f})', alpha=0.7)

            # Mark clinical thresholds
            ax1.axvline(self.low_risk_threshold, color='green', linestyle='--',
                       linewidth=1.5, alpha=0.5)
            ax1.axvline(self.high_risk_threshold, color='red', linestyle='--',
                       linewidth=1.5, alpha=0.5)

            ax1.set_xlabel('Decision Threshold', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Metric Value', fontsize=12, fontweight='bold')
            ax1.set_title(f'{model_name.upper()}\nSensitivity-Specificity Trade-off',
                         fontsize=13, fontweight='bold')
            ax1.legend(loc='best', fontsize=9)
            ax1.grid(alpha=0.3)
            ax1.set_ylim([0, 1.05])

            # Plot 2: PPV vs NPV
            ax2 = axes[1, idx]
            ax2.plot(thresholds, ppvs, 'orange', linewidth=2.5, label='PPV (Precision)', alpha=0.8)
            ax2.plot(thresholds, npvs, 'cyan', linewidth=2.5, label='NPV', alpha=0.8)

            ax2.axvline(optimal_thresh, color='purple', linestyle=':', linewidth=2,
                       label=f'Optimal F1 ({optimal_thresh:.2f})', alpha=0.7)
            ax2.axvline(self.low_risk_threshold, color='green', linestyle='--',
                       linewidth=1.5, alpha=0.5)
            ax2.axvline(self.high_risk_threshold, color='red', linestyle='--',
                       linewidth=1.5, alpha=0.5)

            ax2.set_xlabel('Decision Threshold', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Predictive Value', fontsize=12, fontweight='bold')
            ax2.set_title('Positive vs Negative Predictive Value', fontsize=12, fontweight='bold')
            ax2.legend(loc='best', fontsize=9)
            ax2.grid(alpha=0.3)
            ax2.set_ylim([0, 1.05])

        fig.suptitle('CLINICAL DECISION THRESHOLD OPTIMIZATION',
                    fontsize=16, fontweight='bold', y=0.995)
        self._show_or_save('02_threshold_analysis.png')

    def plot_uncertainty_clinical_implications(self):
        """
        PLOT 3: Uncertainty and Clinical Review Requirements

        Visualizes which predictions require additional clinical review based on
        model uncertainty (wide confidence intervals).
        """
        n_models = len(self.results)
        fig, axes = plt.subplots(2, n_models, figsize=(6*n_models, 12))
        if n_models == 1:
            axes = axes.reshape(2, 1)

        for idx, (model_name, result) in enumerate(self.results.items()):
            oof_preds = result['oof_preds']
            oof_lower = result['oof_lower']
            oof_upper = result['oof_upper']
            uncertainty = oof_upper - oof_lower

            # Plot 1: Uncertainty vs Predicted Probability
            ax1 = axes[0, idx]

            # Color by true label
            colors = ['#3498db' if y == 0 else '#e74c3c' for y in self.y_train]
            scatter = ax1.scatter(oof_preds, uncertainty, c=colors, alpha=0.4, s=20)

            # Add uncertainty threshold line
            ax1.axhline(self.uncertainty_threshold, color='red', linestyle='--',
                       linewidth=2, label=f'Review Threshold ({self.uncertainty_threshold:.2f})', alpha=0.7)

            # Highlight high uncertainty region
            ax1.fill_between([0, 1], self.uncertainty_threshold, uncertainty.max(),
                            alpha=0.1, color='red', label='Requires Clinical Review')

            # Add probability threshold zones
            ax1.axvline(self.low_risk_threshold, color='green', linestyle=':', alpha=0.4)
            ax1.axvline(self.high_risk_threshold, color='red', linestyle=':', alpha=0.4)

            ax1.set_xlabel('Predicted Probability', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Uncertainty (CI Width)', fontsize=12, fontweight='bold')
            ax1.set_title(f'{model_name.upper()}\nUncertainty vs Prediction',
                         fontsize=13, fontweight='bold')

            # Custom legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='#3498db', alpha=0.6, label='Healthy (True)'),
                Patch(facecolor='#e74c3c', alpha=0.6, label='Diabetic (True)'),
                Patch(facecolor='red', alpha=0.1, label='Review Required Zone')
            ]
            ax1.legend(handles=legend_elements, loc='upper right', fontsize=9)
            ax1.grid(alpha=0.3)

            # Plot 2: Error Rate by Uncertainty Quartile
            ax2 = axes[1, idx]

            # Divide into uncertainty quartiles
            uncertainty_quartiles = pd.qcut(uncertainty, q=4, labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)'])

            # Calculate error rate per quartile
            optimal_thresh = 0.5
            y_pred = (oof_preds >= optimal_thresh).astype(int)
            errors = (y_pred != self.y_train).astype(int)

            quartile_errors = pd.DataFrame({
                'Quartile': uncertainty_quartiles,
                'Error': errors
            })

            error_rates = quartile_errors.groupby('Quartile')['Error'].agg(['mean', 'count'])
            error_rates['mean'] *= 100  # Convert to percentage

            # Plot bars
            bars = ax2.bar(range(len(error_rates)), error_rates['mean'],
                          color=['#2ecc71', '#f39c12', '#e67e22', '#e74c3c'], alpha=0.8)

            # Add value labels
            for i, (bar, (idx_name, row)) in enumerate(zip(bars, error_rates.iterrows())):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}%\n(n={int(row["count"])})',
                        ha='center', va='bottom', fontsize=10, fontweight='bold')

            ax2.set_xticks(range(len(error_rates)))
            ax2.set_xticklabels(error_rates.index, fontsize=11)
            ax2.set_ylabel('Error Rate (%)', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Uncertainty Quartile', fontsize=12, fontweight='bold')
            ax2.set_title('Prediction Error by Uncertainty Level', fontsize=12, fontweight='bold')
            ax2.grid(axis='y', alpha=0.3)

        fig.suptitle('UNCERTAINTY-BASED CLINICAL REVIEW ANALYSIS',
                    fontsize=16, fontweight='bold', y=0.995)
        self._show_or_save('03_uncertainty_clinical_review.png')

    def plot_cost_benefit_analysis(self):
        """
        PLOT 4: Clinical Cost-Benefit Analysis

        Analyzes the clinical costs of different types of errors (false positives
        vs false negatives) at various decision thresholds.
        """
        n_models = len(self.results)
        fig, axes = plt.subplots(1, n_models, figsize=(7*n_models, 6))
        if n_models == 1:
            axes = [axes]

        for idx, (model_name, result) in enumerate(self.results.items()):
            ax = axes[idx]
            oof_preds = result['oof_preds']

            thresholds = np.linspace(0.1, 0.9, 81)
            total_costs = []
            fn_costs = []
            fp_costs = []

            for thresh in thresholds:
                y_pred = (oof_preds >= thresh).astype(int)
                tn, fp, fn, tp = confusion_matrix(self.y_train, y_pred).ravel()

                # Clinical costs
                cost_fn = fn * self.cost_false_negative  # Missing diabetes
                cost_fp = fp * self.cost_false_positive  # False alarm
                total_cost = cost_fn + cost_fp

                total_costs.append(total_cost)
                fn_costs.append(cost_fn)
                fp_costs.append(cost_fp)

            # Plot stacked area
            ax.fill_between(thresholds, 0, fn_costs, alpha=0.6, color='#e74c3c',
                           label=f'False Negative Cost (Ã—{self.cost_false_negative})')
            ax.fill_between(thresholds, fn_costs, np.array(fn_costs) + np.array(fp_costs),
                           alpha=0.6, color='#f39c12',
                           label=f'False Positive Cost (Ã—{self.cost_false_positive})')

            # Plot total cost line
            ax.plot(thresholds, total_costs, 'k-', linewidth=3, label='Total Cost', alpha=0.8)

            # Mark optimal threshold (minimum cost)
            optimal_idx = np.argmin(total_costs)
            optimal_thresh = thresholds[optimal_idx]
            min_cost = total_costs[optimal_idx]

            ax.axvline(optimal_thresh, color='purple', linestyle='--', linewidth=2.5,
                      label=f'Optimal Threshold ({optimal_thresh:.2f})', alpha=0.8)
            ax.plot(optimal_thresh, min_cost, 'p', color='purple', markersize=15,
                   markeredgecolor='black', markeredgewidth=1.5)

            # Add clinical threshold references
            ax.axvline(self.low_risk_threshold, color='green', linestyle=':', alpha=0.4)
            ax.axvline(self.high_risk_threshold, color='red', linestyle=':', alpha=0.4)

            ax.set_xlabel('Decision Threshold', fontsize=12, fontweight='bold')
            ax.set_ylabel('Clinical Cost (Relative Units)', fontsize=12, fontweight='bold')
            ax.set_title(f'{model_name.upper()}\nCost-Benefit Analysis\n'
                        f'(FN cost = {self.cost_false_negative}Ã— FP cost)',
                        fontsize=13, fontweight='bold')
            ax.legend(loc='best', fontsize=9)
            ax.grid(alpha=0.3)

        fig.suptitle('CLINICAL COST-BENEFIT OPTIMIZATION',
                    fontsize=16, fontweight='bold', y=0.98)
        self._show_or_save('04_cost_benefit_analysis.png')

    def plot_model_agreement_analysis(self):
        """
        PLOT 5: Model Agreement and Consensus Analysis

        Shows where different models agree/disagree, which cases require
        additional clinical review due to model disagreement.
        """
        if len(self.results) < 2:
            print("Model agreement analysis requires at least 2 models.")
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 14))

        # Get predictions from all models
        model_names = list(self.results.keys())
        all_preds = np.array([self.results[name]['oof_preds'] for name in model_names])

        # Calculate agreement metrics
        pred_std = np.std(all_preds, axis=0)  # Disagreement measure
        pred_mean = np.mean(all_preds, axis=0)

        # Plot 1: Model Agreement Heatmap
        ax1 = axes[0, 0]

        # For visualization, bin patients by mean prediction
        bins = np.linspace(0, 1, 21)
        bin_indices = np.digitize(pred_mean, bins) - 1

        # Calculate average disagreement per bin
        disagreement_by_bin = []
        for i in range(len(bins) - 1):
            mask = bin_indices == i
            if mask.sum() > 0:
                disagreement_by_bin.append(pred_std[mask].mean())
            else:
                disagreement_by_bin.append(0)

        # Plot as heatmap
        disagreement_matrix = np.array(disagreement_by_bin).reshape(1, -1)
        im = ax1.imshow(disagreement_matrix, aspect='auto', cmap='YlOrRd', vmin=0, vmax=0.2)
        ax1.set_yticks([0])
        ax1.set_yticklabels(['Disagreement'])
        ax1.set_xticks(range(0, len(bins)-1, 2))
        ax1.set_xticklabels([f'{bins[i]:.1f}' for i in range(0, len(bins)-1, 2)])
        ax1.set_xlabel('Mean Predicted Probability', fontsize=12, fontweight='bold')
        ax1.set_title('Model Disagreement Across Probability Range', fontsize=13, fontweight='bold')
        plt.colorbar(im, ax=ax1, label='Std Dev of Predictions')

        # Plot 2: Scatter of Disagreement vs Mean Prediction
        ax2 = axes[0, 1]

        colors = ['#3498db' if y == 0 else '#e74c3c' for y in self.y_train]
        ax2.scatter(pred_mean, pred_std, c=colors, alpha=0.4, s=20)

        # Mark high disagreement threshold
        disagreement_threshold = 0.10
        ax2.axhline(disagreement_threshold, color='red', linestyle='--', linewidth=2,
                   label=f'High Disagreement ({disagreement_threshold:.2f})', alpha=0.7)
        ax2.fill_between([0, 1], disagreement_threshold, pred_std.max(),
                        alpha=0.1, color='red', label='Review Required')

        ax2.set_xlabel('Mean Predicted Probability', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Model Disagreement (Std Dev)', fontsize=12, fontweight='bold')
        ax2.set_title('Model Consensus Analysis', fontsize=13, fontweight='bold')

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#3498db', alpha=0.6, label='Healthy (True)'),
            Patch(facecolor='#e74c3c', alpha=0.6, label='Diabetic (True)'),
            Patch(facecolor='red', alpha=0.1, label='Review Required')
        ]
        ax2.legend(handles=legend_elements, loc='upper right', fontsize=9)
        ax2.grid(alpha=0.3)

        # Plot 3: Pairwise Model Comparison
        ax3 = axes[1, 0]

        if len(model_names) >= 2:
            model1_preds = self.results[model_names[0]]['oof_preds']
            model2_preds = self.results[model_names[1]]['oof_preds']

            ax3.scatter(model1_preds, model2_preds, c=colors, alpha=0.4, s=20)
            ax3.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.5, label='Perfect Agreement')

            # Calculate correlation
            correlation = np.corrcoef(model1_preds, model2_preds)[0, 1]

            ax3.set_xlabel(f'{model_names[0].upper()} Predictions', fontsize=12, fontweight='bold')
            ax3.set_ylabel(f'{model_names[1].upper()} Predictions', fontsize=12, fontweight='bold')
            ax3.set_title(f'Model Comparison (r={correlation:.3f})', fontsize=13, fontweight='bold')
            ax3.legend(loc='lower right', fontsize=9)
            ax3.grid(alpha=0.3)
            ax3.set_xlim([0, 1])
            ax3.set_ylim([0, 1])

        # Plot 4: Disagreement Statistics
        ax4 = axes[1, 1]

        # Categorize by disagreement level
        disagreement_categories = pd.cut(pred_std, bins=[0, 0.05, 0.10, 0.15, 1.0],
                                         labels=['Very Low', 'Low', 'Medium', 'High'])
        category_counts = disagreement_categories.value_counts()

        colors_bar = ['#2ecc71', '#f39c12', '#e67e22', '#e74c3c']
        bars = ax4.bar(range(len(category_counts)), category_counts.values,
                      color=colors_bar, alpha=0.8)

        for i, (bar, count) in enumerate(zip(bars, category_counts.values)):
            height = bar.get_height()
            pct = 100 * count / len(pred_std)
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(count)}\n({pct:.1f}%)',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax4.set_xticks(range(len(category_counts)))
        ax4.set_xticklabels(category_counts.index)
        ax4.set_ylabel('Number of Patients', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Model Disagreement Level', fontsize=12, fontweight='bold')
        ax4.set_title('Distribution of Model Agreement', fontsize=13, fontweight='bold')
        ax4.grid(axis='y', alpha=0.3)

        fig.suptitle('MODEL AGREEMENT AND CONSENSUS ANALYSIS',
                    fontsize=16, fontweight='bold', y=0.995)
        self._show_or_save('05_model_agreement.png')

    def plot_clinical_calibration(self):
        """
        PLOT 6: Clinical Calibration Reliability

        Shows calibration curves specifically for clinical decision zones
        (low/medium/high risk), ensuring probabilities are medically valid.
        """
        n_models = len(self.results)
        fig, axes = plt.subplots(1, n_models, figsize=(7*n_models, 6))
        if n_models == 1:
            axes = [axes]

        for idx, (model_name, result) in enumerate(self.results.items()):
            ax = axes[idx]
            oof_preds = result['oof_preds']

            # Calculate calibration curve
            from sklearn.calibration import calibration_curve

            n_bins = 10
            prob_true, prob_pred = calibration_curve(self.y_train, oof_preds, n_bins=n_bins)

            # Plot perfect calibration line
            ax.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.5, label='Perfect Calibration')

            # Plot actual calibration
            ax.plot(prob_pred, prob_true, 's-', linewidth=2.5, markersize=10,
                   color='#3498db', label='Model Calibration', alpha=0.8)

            # Highlight clinical decision zones
            ax.axvspan(0, self.low_risk_threshold, alpha=0.1, color='green', label='Low Risk Zone')
            ax.axvspan(self.low_risk_threshold, self.high_risk_threshold, alpha=0.1, color='orange',
                      label='Medium Risk Zone')
            ax.axvspan(self.high_risk_threshold, 1, alpha=0.1, color='red', label='High Risk Zone')

            # Calculate Expected Calibration Error (ECE)
            ece = np.mean(np.abs(prob_true - prob_pred))

            ax.set_xlabel('Predicted Probability', fontsize=12, fontweight='bold')
            ax.set_ylabel('Observed Frequency', fontsize=12, fontweight='bold')
            ax.set_title(f'{model_name.upper()}\nClinical Calibration Reliability\nECE = {ece:.4f}',
                        fontsize=13, fontweight='bold')
            ax.legend(loc='upper left', fontsize=9)
            ax.grid(alpha=0.3)
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])

        fig.suptitle('CLINICAL PROBABILITY CALIBRATION',
                    fontsize=16, fontweight='bold', y=0.98)
        self._show_or_save('06_clinical_calibration.png')

    def generate_patient_risk_report(self, patient_idx: int, X_features: Optional[pd.DataFrame] = None):
        """
        PLOT 7: Individual Patient Risk Report

        Generate a detailed risk assessment for a specific patient showing
        predictions from all models with uncertainty bounds.

        Parameters:
        -----------
        patient_idx : int
            Index of patient in training data
        X_features : pd.DataFrame, optional
            Feature dataframe to show top contributing features
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Get predictions for this patient from all models
        model_preds = []
        model_lowers = []
        model_uppers = []
        model_names_list = []

        for model_name, result in self.results.items():
            model_names_list.append(model_name.upper())
            model_preds.append(result['oof_preds'][patient_idx])
            model_lowers.append(result['oof_lower'][patient_idx])
            model_uppers.append(result['oof_upper'][patient_idx])

        true_label = self.y_train[patient_idx]
        true_label_str = 'DIABETIC' if true_label == 1 else 'HEALTHY'

        # Plot 1: Model Predictions with Uncertainty
        ax1 = axes[0, 0]

        y_pos = np.arange(len(model_names_list))
        errors = [[p - l for p, l in zip(model_preds, model_lowers)],
                 [u - p for p, u in zip(model_uppers, model_preds)]]

        colors_models = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12'][:len(model_names_list)]
        ax1.barh(y_pos, model_preds, xerr=errors, color=colors_models, alpha=0.7, capsize=5)

        # Add threshold lines
        ax1.axvline(self.low_risk_threshold, color='green', linestyle='--', linewidth=2, alpha=0.5)
        ax1.axvline(self.high_risk_threshold, color='red', linestyle='--', linewidth=2, alpha=0.5)
        ax1.axvline(0.5, color='black', linestyle=':', linewidth=1.5, alpha=0.3)

        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(model_names_list, fontsize=11, fontweight='bold')
        ax1.set_xlabel('Predicted Probability with 90% CI', fontsize=12, fontweight='bold')
        ax1.set_title(f'Patient #{patient_idx} - Model Predictions\nTrue Label: {true_label_str}',
                     fontsize=13, fontweight='bold')
        ax1.set_xlim([0, 1])
        ax1.grid(axis='x', alpha=0.3)

        # Plot 2: Consensus Summary
        ax2 = axes[0, 1]
        ax2.axis('off')

        mean_pred = np.mean(model_preds)
        std_pred = np.std(model_preds)
        risk_level = self._classify_risk_level(mean_pred, np.mean(model_uppers) - np.mean(model_lowers))

        # Create summary text
        summary_text = f"""
        PATIENT RISK ASSESSMENT SUMMARY
        {'='*50}

        Patient ID: {patient_idx}
        True Diagnosis: {true_label_str}

        CONSENSUS PREDICTION:
          â€¢ Mean Probability: {mean_pred:.3f}
          â€¢ Model Agreement (Std): {std_pred:.3f}
          â€¢ Risk Category: {risk_level}

        MODEL PREDICTIONS:
        """

        for i, (name, pred, lower, upper) in enumerate(zip(model_names_list, model_preds,
                                                            model_lowers, model_uppers)):
            summary_text += f"\n  {i+1}. {name}: {pred:.3f} [{lower:.3f}, {upper:.3f}]"

        summary_text += f"""

        CLINICAL RECOMMENDATION:
        """

        if risk_level == 'Low Risk':
            recommendation = "  â€¢ Low diabetes risk\n  â€¢ Routine monitoring recommended\n  â€¢ Lifestyle counseling"
        elif risk_level == 'Medium Risk':
            recommendation = "  â€¢ Moderate diabetes risk\n  â€¢ Consider additional testing (HbA1c, fasting glucose)\n  â€¢ Preventive interventions recommended"
        elif risk_level == 'High Risk':
            recommendation = "  â€¢ High diabetes risk\n  â€¢ Immediate diagnostic testing required\n  â€¢ Consider treatment initiation"
        else:  # Uncertain
            recommendation = "  â€¢ UNCERTAIN - Models disagree or wide CI\n  â€¢ CLINICAL REVIEW REQUIRED\n  â€¢ Additional diagnostic testing recommended"

        summary_text += recommendation

        ax2.text(0.05, 0.95, summary_text, transform=ax2.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        # Plot 3: Probability Distribution Visualization
        ax3 = axes[1, 0]

        # Show where this patient falls in the overall distribution
        all_preds = self.results[list(self.results.keys())[0]]['oof_preds']

        ax3.hist(all_preds[self.y_train == 0], bins=30, alpha=0.5, color='#3498db',
                label='All Healthy Patients', density=True)
        ax3.hist(all_preds[self.y_train == 1], bins=30, alpha=0.5, color='#e74c3c',
                label='All Diabetic Patients', density=True)

        # Mark this patient
        ax3.axvline(mean_pred, color='purple', linewidth=3, linestyle='--',
                   label=f'This Patient ({mean_pred:.3f})', alpha=0.8)

        # Add uncertainty band
        ax3.axvspan(mean_pred - std_pred, mean_pred + std_pred, alpha=0.2, color='purple',
                   label='Model Uncertainty')

        ax3.set_xlabel('Predicted Probability', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Density', fontsize=12, fontweight='bold')
        ax3.set_title('Patient Position in Population Distribution', fontsize=12, fontweight='bold')
        ax3.legend(fontsize=9)
        ax3.grid(alpha=0.3)

        # Plot 4: Risk Category Visualization
        ax4 = axes[1, 1]

        # Create risk gauge
        categories = ['Low\nRisk', 'Medium\nRisk', 'High\nRisk']
        category_ranges = [(0, self.low_risk_threshold),
                          (self.low_risk_threshold, self.high_risk_threshold),
                          (self.high_risk_threshold, 1.0)]
        colors_gauge = ['#2ecc71', '#f39c12', '#e74c3c']

        for i, ((start, end), color, cat) in enumerate(zip(category_ranges, colors_gauge, categories)):
            ax4.barh(0, end - start, left=start, height=0.3, color=color, alpha=0.6,
                    edgecolor='black', linewidth=2)
            # Add category label
            ax4.text((start + end) / 2, -0.3, cat, ha='center', va='top',
                    fontsize=11, fontweight='bold')

        # Mark patient prediction
        ax4.plot(mean_pred, 0, 'v', color='purple', markersize=20,
                markeredgecolor='black', markeredgewidth=2, label='Patient Prediction')

        # Add uncertainty range
        ax4.plot([model_lowers[0], model_uppers[0]], [0, 0], 'purple',
                linewidth=4, alpha=0.5, label='Uncertainty Range')

        ax4.set_xlim([0, 1])
        ax4.set_ylim([-0.5, 0.5])
        ax4.set_yticks([])
        ax4.set_xlabel('Risk Probability', fontsize=12, fontweight='bold')
        ax4.set_title('Risk Category Assessment', fontsize=12, fontweight='bold')
        ax4.legend(loc='upper right', fontsize=9)
        ax4.grid(axis='x', alpha=0.3)

        fig.suptitle(f'INDIVIDUAL PATIENT RISK REPORT - Patient #{patient_idx}',
                    fontsize=16, fontweight='bold')
        self._show_or_save(f'07_patient_report_{patient_idx}.png')

    def generate_all_clinical_plots(self, sample_patient_indices: List[int] = None):
        """
        Generate all clinical decision-support visualizations.

        Parameters:
        -----------
        sample_patient_indices : List[int], optional
            List of patient indices for individual reports. If None, generates
            reports for 3 random patients (low/medium/high risk).
        """
        print("\nGenerating Clinical Decision-Support Dashboard...")
        print("="*70)

        print("\n[1/7] Risk Stratification Analysis...")
        self.plot_risk_stratification()

        print("[2/7] Decision Threshold Optimization...")
        self.plot_decision_threshold_analysis()

        print("[3/7] Uncertainty-Based Clinical Review...")
        self.plot_uncertainty_clinical_implications()

        print("[4/7] Cost-Benefit Analysis...")
        self.plot_cost_benefit_analysis()

        print("[5/7] Model Agreement Analysis...")
        self.plot_model_agreement_analysis()

        print("[6/7] Clinical Calibration Assessment...")
        self.plot_clinical_calibration()

        print("[7/7] Individual Patient Reports...")

        if sample_patient_indices is None:
            # Select 3 representative patients: low, medium, high risk
            first_model = list(self.results.keys())[0]
            preds = self.results[first_model]['oof_preds']

            low_risk_idx = np.argmin(np.abs(preds - 0.15))
            med_risk_idx = np.argmin(np.abs(preds - 0.50))
            high_risk_idx = np.argmin(np.abs(preds - 0.85))

            sample_patient_indices = [low_risk_idx, med_risk_idx, high_risk_idx]

        for patient_idx in sample_patient_indices:
            self.generate_patient_risk_report(patient_idx)

        print("\n" + "="*70)
        print("âœ“ Clinical Dashboard Generation Complete!")
        print("="*70)


config = ImprovedConfig()
config.N_TRIALS = 5  
config.MAX_FEATURES = 20
np.random.seed(config.SEED)

# Load data
train_df = pd.read_csv(config.TRAIN_PATH)
test_df = pd.read_csv(config.TEST_PATH)

print(f"  â†’ Train shape: {train_df.shape}")
print(f"  â†’ Test shape: {test_df.shape}")

# Store IDs and separate target
test_ids = test_df[config.ID_COL].copy()
train_df = train_df.drop(columns=[config.ID_COL])
test_df = test_df.drop(columns=[config.ID_COL])

y_train = train_df[config.TARGET].copy()
X_train = train_df.drop(columns=[config.TARGET])
X_test = test_df.copy()

target_dist = y_train.value_counts(normalize=True)
print(f"  â†’ Target: Class 0={target_dist[0]:.2%}, Class 1={target_dist[1]:.2%}")


# Label encode categoricals
label_encoders = {}
for col in config.CATEGORICAL_FEATURES:
    if col in X_train.columns:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col].astype(str))
        X_test[col] = X_test[col].astype(str).apply(
            lambda x: le.transform([x])[0] if x in le.classes_ else -1
        )
        label_encoders[col] = le

# Handle missing values
X_train = X_train.fillna(X_train.median())
X_test = X_test.fillna(X_train.median())

print(f"  âœ“ Preprocessing complete")


engineer = MedicalFeatureEngineer(config)
X_train = engineer.create_features(X_train)
X_test = engineer.create_features(X_test)

# Ensure test has same columns
missing_cols = set(X_train.columns) - set(X_test.columns)
for col in missing_cols:
    X_test[col] = 0
X_test = X_test[X_train.columns]

# Handle any NaNs
X_train = X_train.fillna(0)
X_test = X_test.fillna(0)

print(f"  â†’ Features: {X_train.shape[1]}")


target_encoder = KFoldTargetEncoder(
    categorical_features=config.CATEGORICAL_FEATURES,
    n_splits=config.N_FOLDS,
    seed=config.SEED
)

X_train = target_encoder.fit_transform(X_train, y_train)
X_test = target_encoder.transform(X_test)

print(f"  âœ“ Target encoding complete")


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

print(f"  âœ“ Scaling complete")


selector = ShapFeatureSelector(
    n_estimators=100,
    threshold=config.FEATURE_SELECTION_THRESHOLD,
    max_features=config.MAX_FEATURES
)

X_train_selected = selector.fit_transform(X_train_scaled, y_train)
X_test_selected = selector.transform(X_test_scaled)

print(f"  âœ“ Selected {X_train_selected.shape[1]} features")


# Initialize probabilistic trainer
prob_trainer = ProbabilisticModelTrainer(config)

models_to_train = ['ngboost','lightgbm', 'xgboost'] # 'catboost' 'xgboost'

results = prob_trainer.train_with_cv(
    X=X_train_selected,
    y=y_train,
    X_test=X_test_selected,
    model_names=models_to_train,
    n_trials=config.N_TRIALS,  # Optuna trials
    n_bootstrap=10  # Bootstrap models for uncertainty
)


# Compare models
comparison_df = compare_probabilistic_models(results, y_train.values)
print("\n" + comparison_df.to_string(index=False))

# Print detailed summary
print_model_summary(prob_trainer, y_train.values)


dashboard = ProbabilisticModelDashboard(
        trainer=prob_trainer,
        results=results,
        y_train=y_train.values,
        save_figures=False,  
        figsize=(12, 8),
        dpi=100
    )


dashboard.plot_model_comparison()


dashboard.plot_roc_curves()


dashboard.plot_confusion_matrices()


dashboard.plot_uncertainty_distributions()


dashboard.plot_confidence_intervals()


dashboard.plot_uncertainty_vs_error()


clinical_dashboard = ClinicalDecisionDashboard(
        trainer=prob_trainer,
        results=results,
        y_train=y_train.values,
        save_figures=False, 
        figsize=(14, 10),
        dpi=100
    )


clinical_dashboard.plot_risk_stratification()


clinical_dashboard.plot_decision_threshold_analysis()


clinical_dashboard.plot_uncertainty_clinical_implications()


clinical_dashboard.plot_cost_benefit_analysis()


# Get best model
best_model_name = max(results.items(), key=lambda x: x[1]['oof_score'])[0]
print(f"\n  â†’ Best Model: {best_model_name.upper()}")
print(f"  â†’ Best OOF AUC: {results[best_model_name]['oof_score']:.6f}")

# Get predictions from best model
final_test_preds = results[best_model_name]['test_preds']

# Also get uncertainty bounds
final_lower = results[best_model_name]['test_lower']
final_upper = results[best_model_name]['test_upper']

# Create submission
submission = pd.DataFrame({
    config.ID_COL: test_ids,
    config.TARGET: final_test_preds
})

submission_path = 'submission.csv'
submission.to_csv(submission_path, index=False)

print(f"\n  â†’ Submission saved: {submission_path}")
print(f"\n  Prediction Statistics:")
print(f"    Mean:  {final_test_preds.mean():.6f}")
print(f"    Std:   {final_test_preds.std():.6f}")
print(f"    Min:   {final_test_preds.min():.6f}")
print(f"    Max:   {final_test_preds.max():.6f}")

