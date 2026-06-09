## ULTIMATE KAGGLE ENSEMBLE: MEDICAL RISK EDITION

import sys, subprocess, time, warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. SETUP & SILENCE WARNINGS
# ------------------------------------------------------------------------------
warnings.filterwarnings('ignore', category=SyntaxWarning)
warnings.filterwarnings('ignore')

def install(package):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])
    except:
        pass

print("--- Preparing High-Performance Environment ---")
install('lightgbm')
install('catboost')
install('xgboost')

import lightgbm as lgb
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

# Paths
BASE_PATH = "/kaggle/input/playground-series-s5e12"
TRAIN_FILE = f"{BASE_PATH}/train.csv"
TEST_FILE = f"{BASE_PATH}/test.csv"

# 2. ADVANCED MEDICAL FEATURE ENGINEERING
# ------------------------------------------------------------------------------
print("--- Engineering Medical Interaction Features ---")
train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

y = train_df['diagnosed_diabetes']
test_ids = test_df['id']
X = train_df.drop(['id', 'diagnosed_diabetes'], axis=1)
X_test = test_df.drop('id', axis=1)

def feature_engineer(df):
    # Blood Pressure Stats
    df['mean_bp'] = (df['systolic_bp'] + df['diastolic_bp']) / 2
    df['pulse_pressure'] = df['systolic_bp'] - df['diastolic_bp']
    
    # Ratios (Crucial for Diabetes)
    df['chol_ratio_hdl_ldl'] = df['hdl_cholesterol'] / (df['ldl_cholesterol'] + 1e-6)
    df['hdl_to_total_ratio'] = df['hdl_cholesterol'] / (df['cholesterol_total'] + 1e-6)
    
    # Medical Risk Interactions
    # Metabolic Index: Combines high BP and high BMI
    df['metabolic_index'] = (df['mean_bp'] * df['bmi']) / 100
    
    # Health Risk Score: Sum of existing conditions
    df['health_risk_score'] = (df['family_history_diabetes'] + 
                               df['hypertension_history'] + 
                               df['cardiovascular_history'])
    
    # Lifestyle Impact
    df['age_bmi_prod'] = df['age'] * df['bmi']
    df['sedentary_ratio'] = df['screen_time_hours_per_day'] / (df['physical_activity_minutes_per_week'] / 60 + 1)
    
    # Categorical Clean
    df['income_level'] = df['income_level'].str.replace('-', '_', regex=False)
    return df

X = feature_engineer(X)
X_test = feature_engineer(X_test)

# 3. ROBUST PREPROCESSING
# ------------------------------------------------------------------------------
num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = X.select_dtypes(include=['object']).columns.tolist()
bin_cols = ['family_history_diabetes', 'hypertension_history', 'cardiovascular_history', 'health_risk_score']
num_cols = [c for c in num_cols if c not in bin_cols]

preprocessor = ColumnTransformer(transformers=[
    ('num', Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler())]), num_cols),
    ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), cat_cols),
    ('bin', 'passthrough', bin_cols)
])

X_proc = preprocessor.fit_transform(X)
X_test_proc = preprocessor.transform(X_test)
feat_names = num_cols + preprocessor.named_transformers_['cat'].get_feature_names_out(cat_cols).tolist() + bin_cols

# 4. TRAINING THE TRIPLE ENSEMBLE (5-FOLD)
# ------------------------------------------------------------------------------
print(f"\n--- Training Triple Ensemble ({X_proc.shape[1]} Features) ---")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_preds = np.zeros(len(X))
final_test_preds = np.zeros(len(X_test))
importances = np.zeros(len(feat_names))

for fold, (t_idx, v_idx) in enumerate(skf.split(X_proc, y)):
    X_tr, X_va = X_proc[t_idx], X_proc[v_idx]
    y_tr, y_va = y.iloc[t_idx], y.iloc[v_idx]
    
    # Model 1: LightGBM (Weighted 45%) - Optimized for non-linear depth
    m_lgb = lgb.LGBMClassifier(n_estimators=2000, learning_rate=0.02, max_depth=10, num_leaves=63, verbose=-1, random_state=42)
    m_lgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(100, verbose=False)])
    
    # Model 2: CatBoost (Weighted 35%) - Great for categorical stability
    m_cat = CatBoostClassifier(iterations=2000, learning_rate=0.02, depth=7, verbose=0, random_seed=42, early_stopping_rounds=100)
    m_cat.fit(X_tr, y_tr, eval_set=(X_va, y_va))
    
    # Model 3: XGBoost (Weighted 20%) - Provides structural diversity
    m_xgb = XGBClassifier(n_estimators=2000, learning_rate=0.02, max_depth=7, random_state=42, tree_method='hist', early_stopping_rounds=100)
    m_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    
    # Blending the Fold
    p_lgb = m_lgb.predict_proba(X_test_proc)[:, 1]
    p_cat = m_cat.predict_proba(X_test_proc)[:, 1]
    p_xgb = m_xgb.predict_proba(X_test_proc)[:, 1]
    
    # Using 45/35/20 split for better ensemble balance
    final_test_preds += ((p_lgb * 0.45) + (p_cat * 0.35) + (p_xgb * 0.20)) / 5
    
    # Track CV Score for this fold
    val_blend = (m_lgb.predict_proba(X_va)[:, 1] * 0.45) + (m_cat.predict_proba(X_va)[:, 1] * 0.35) + (m_xgb.predict_proba(X_va)[:, 1] * 0.20)
    print(f"Fold {fold+1} Blended AUC: {roc_auc_score(y_va, val_blend):.5f}")
    
    importances += m_lgb.feature_importances_ / 5

# 5. RESULTS & SUBMISSION
# ------------------------------------------------------------------------------
# Feature Importance Visualization
plt.figure(figsize=(10, 6))
imp_df = pd.DataFrame({'feat': feat_names, 'imp': importances}).sort_values('imp', ascending=False).head(12)
sns.barplot(data=imp_df, x='imp', y='feat', palette='magma')
plt.title('Top 12 Drivers of Diabetes Prediction')
plt.show()

# Create Submission
pd.DataFrame({'id': test_ids, 'diagnosed_diabetes': final_test_preds}).to_csv('submission.csv', index=False)
print("✓ submission.csv is ready.")

