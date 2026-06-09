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


import os
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import optuna
import warnings

# Configuration
warnings.filterwarnings('ignore')
OPTUNA_TRIALS = 50  # Increased for better optimization
N_FOLDS = 10  # 10-fold CV for better validation
SEED = 42



# ======================================================================
# IMPORTS & CONFIGURATION
# ======================================================================

import os
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
import optuna
import warnings

# Configuration
warnings.filterwarnings('ignore')
OPTUNA_TRIALS = 50  # Increased for better optimization
N_FOLDS = 10  # 10-fold CV for better validation
SEED = 42

# ======================================================================
# 1. ADVANCED MEDICAL FEATURE ENGINEERING
# ======================================================================

def engineer_medical_features(df):
    """
    Derives physiological indices based on medical standards.
    Handles unit conversions and interactions.
    """
    df_eng = df.copy()

    # --- A. Lipid Profile Transformations ---
    # Conversion factors: mg/dL -> mmol/L
    # TC, LDL, HDL factor: 0.02586
    # TG factor: 0.01129

    tc_mmol = df_eng['cholesterol_total'] * 0.02586
    hdl_mmol = df_eng['hdl_cholesterol'] * 0.02586
    ldl_mmol = df_eng['ldl_cholesterol'] * 0.02586
    tg_mmol = df_eng['triglycerides'] * 0.01129

    # 1. Atherogenic Index of Plasma (AIP)
    # Formula: log10(TG / HDL) in molar units
    # Small shift +1e-5 to avoid log(0)
    df_eng['AIP'] = np.log10((tg_mmol + 1e-5) / (hdl_mmol + 1e-5))

    # 2. Castelli Risk Indices (CRI)
    df_eng['CRI_1'] = df_eng['cholesterol_total'] / df_eng['hdl_cholesterol']
    df_eng['CRI_2'] = df_eng['ldl_cholesterol'] / df_eng['hdl_cholesterol']

    # 3. Atherogenic Coefficient (AC)
    df_eng['AC'] = (df_eng['cholesterol_total'] - df_eng['hdl_cholesterol']) / df_eng['hdl_cholesterol']

    # 4. Non-HDL Cholesterol (proxy for atherogenic lipoproteins)
    df_eng['Non_HDL'] = df_eng['cholesterol_total'] - df_eng['hdl_cholesterol']

    # --- B. Hemodynamic Indices ---

    # 5. Pulse Pressure (PP)
    df_eng['Pulse_Pressure'] = df_eng['systolic_bp'] - df_eng['diastolic_bp']

    # 6. Mean Arterial Pressure (MAP)
    df_eng['MAP'] = df_eng['diastolic_bp'] + (1/3) * df_eng['Pulse_Pressure']

    # 7. Rate Pressure Product (RPP) - myocardial oxygen demand proxy
    df_eng['RPP'] = df_eng['heart_rate'] * df_eng['systolic_bp']

    # --- C. Anthropometric Interactions ---

    # 8. Visceral Adiposity Proxy
    if 'waist_to_hip_ratio' in df_eng.columns:
        df_eng['Visceral_Proxy'] = df_eng['bmi'] * df_eng['waist_to_hip_ratio']
    else:
        df_eng['Visceral_Proxy'] = df_eng['bmi'] ** 1.2

    # 9. Body Shape Index (ABSI) approximation / Conicity Proxy
    df_eng['Conicity_Proxy'] = df_eng['bmi'] * (df_eng['bmi'] / 25) ** 0.5

    return df_eng


# ======================================================================
# 2. DATA PIPELINE
# ======================================================================

def preprocess_data(train_path, test_path):
    """
    Comprehensive data preprocessing and feature engineering pipeline.
    """
    print("Loading data...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Drop ID columns
    train_ids = train_df['id']
    test_ids = test_df['id']
    train_df = train_df.drop('id', axis=1)
    test_df = test_df.drop('id', axis=1)

    # Extract target variable
    target = train_df['diagnosed_diabetes']
    train_df = train_df.drop('diagnosed_diabetes', axis=1)

    # --- Feature Engineering ---
    print("Applying feature engineering...")
    train_df = engineer_medical_features(train_df)
    test_df = engineer_medical_features(test_df)

    # --- Handle Categorical Variables ---
    print("Processing categorical features...")
    cat_cols = [col for col in train_df.columns if train_df[col].dtype == 'object']

    # Label Encoding for Tree Models
    le_dict = {}
    for col in cat_cols:
        le = LabelEncoder()
        # Combine train and test for consistent encoding
        all_data = pd.concat([train_df[col], test_df[col]])
        le.fit(all_data)
        le_dict[col] = le
        train_df[col] = le.transform(train_df[col])
        test_df[col] = le.transform(test_df[col])

    # --- Handle Missing Values ---
    print("Handling missing values...")
    imputer = SimpleImputer(strategy='mean')
    train_df = pd.DataFrame(
        imputer.fit_transform(train_df),
        columns=train_df.columns
    )
    test_df = pd.DataFrame(
        imputer.transform(test_df),
        columns=test_df.columns
    )

    print(f"Training set shape: {train_df.shape}")
    print(f"Test set shape: {test_df.shape}")
    print(f"Target distribution:\n{target.value_counts()}")

    return train_df, target, test_df, train_ids, test_ids, cat_cols


# ======================================================================
# 3. OPTUNA OBJECTIVE FUNCTIONS
# ======================================================================

def objective_xgb(trial, X, y):
    """
    XGBoost hyperparameter optimization objective.
    """
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
        'gamma': trial.suggest_float('gamma', 0.0, 5.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
        'tree_method': 'hist',
        'eval_metric': 'logloss',  # FIXED: Changed from 'auc'
        'random_state': SEED
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_scores = []

    for train_idx, val_idx in cv.split(X, y):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        model = xgb.XGBClassifier(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=50,
            verbose=False
        )

        preds = model.predict_proba(X_val)[:, 1]
        cv_scores.append(roc_auc_score(y_val, preds))

    return np.mean(cv_scores)


def objective_lgbm(trial, X, y):
    """
    LightGBM hyperparameter optimization objective.
    """
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 30),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
        'metric': 'auc',
        'verbosity': -1,
        'random_state': SEED
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_scores = []

    for train_idx, val_idx in cv.split(X, y):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50)],  # FIXED: Wrapped in list
            verbose=False
        )

        preds = model.predict_proba(X_val)[:, 1]
        cv_scores.append(roc_auc_score(y_val, preds))

    return np.mean(cv_scores)


# ======================================================================
# 4. TRAINING AND STACKING
# ======================================================================

def train_full_stack(X, y, X_test, best_xgb_params, best_lgbm_params):
    """
    Trains Level-0 models using 10-Fold Stratified CV.
    Returns OOF predictions and test predictions for meta-learning.
    """
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    # Placeholders for OOF and Test predictions
    oof_xgb = np.zeros(len(X))
    oof_lgbm = np.zeros(len(X))
    oof_cat = np.zeros(len(X))

    test_xgb = np.zeros(len(X_test))
    test_lgbm = np.zeros(len(X_test))
    test_cat = np.zeros(len(X_test))

    # --- Level 0 Training ---
    print(f"Starting {N_FOLDS}-Fold Stratified CV...")

    # CatBoost Params (manual tuning)
    cat_params = {
        'iterations': 2000,
        'learning_rate': 0.03,
        'depth': 6,
        'l2_leaf_reg': 3,
        'loss_function': 'Logloss',
        'eval_metric': 'AUC',
        'verbose': False,
        'random_seed': SEED,
        'allow_writing_files': False
    }

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"Processing Fold {fold + 1}/{N_FOLDS}...")

        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        # 1. XGBoost
        model_xgb = xgb.XGBClassifier(**best_xgb_params)
        model_xgb.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=50,
            verbose=False
        )
        oof_xgb[val_idx] = model_xgb.predict_proba(X_val)[:, 1]
        test_xgb += model_xgb.predict_proba(X_test)[:, 1]

        # 2. LightGBM
        model_lgbm = lgb.LGBMClassifier(**best_lgbm_params)
        model_lgbm.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50)],  # FIXED: Wrapped in list
            verbose=False
        )
        oof_lgbm[val_idx] = model_lgbm.predict_proba(X_val)[:, 1]
        test_lgbm += model_lgbm.predict_proba(X_test)[:, 1]

        # 3. CatBoost
        model_cat = CatBoostClassifier(**cat_params)
        model_cat.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])
        oof_cat[val_idx] = model_cat.predict_proba(X_val)[:, 1]
        test_cat += model_cat.predict_proba(X_test)[:, 1]

    # Average test predictions across folds
    test_xgb /= N_FOLDS
    test_lgbm /= N_FOLDS
    test_cat /= N_FOLDS

    # --- Scores ---
    print(f"\nXGBoost OOF AUC: {roc_auc_score(y, oof_xgb):.4f}")
    print(f"LightGBM OOF AUC: {roc_auc_score(y, oof_lgbm):.4f}")
    print(f"CatBoost OOF AUC: {roc_auc_score(y, oof_cat):.4f}")

    # --- Level 1 Stacking (Meta-Features) ---
    X_meta = pd.DataFrame({
        'xgb': oof_xgb,
        'lgbm': oof_lgbm,
        'cat': oof_cat
    })

    X_meta_test = pd.DataFrame({
        'xgb': test_xgb,
        'lgbm': test_lgbm,
        'cat': test_cat
    })

    print("\nTraining Meta-Learner (Logistic Regression)...")
    meta_model = LogisticRegression(solver='liblinear', random_state=SEED)
    meta_model.fit(X_meta, y)

    final_predictions = meta_model.predict_proba(X_meta_test)[:, 1]

    print(f"Meta-Learner OOF AUC: {roc_auc_score(y, meta_model.predict_proba(X_meta)[:, 1]):.4f}")

    return final_predictions


