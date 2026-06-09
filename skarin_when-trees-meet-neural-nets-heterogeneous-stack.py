# --- 1. SETUP & IMPORTS ---
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import lightgbm as lgb
import xgboost as xgb
import warnings
import gc
import os

warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
CONFIG = {
    'SEED': 42,
    'FOLDS': 10,
    'TARGET': 'diagnosed_diabetes',
    'TRAIN_PATH': '/kaggle/input/playground-series-s5e12/train.csv',
    'TEST_PATH': '/kaggle/input/playground-series-s5e12/test.csv',
    # USING BORUTA FEATURE SET
    'FEATURES': [
        'age', 'physical_activity_minutes_per_week', 'diet_score', 
        'screen_time_hours_per_day', 'bmi', 'waist_to_hip_ratio', 
        'systolic_bp', 'diastolic_bp', 'heart_rate', 'cholesterol_total', 
        'hdl_cholesterol', 'ldl_cholesterol', 'triglycerides', 
        'family_history_diabetes', 'hypertension_history', 
        'cardiovascular_history', 'Age_x_BMI', 'Age_x_BP', 
        'Visceral_Fat_Proxy', 'Sedentary_Index', 'Healthy_Lifestyle_Score'
    ]
}

def set_seed(seed=42):
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(CONFIG['SEED'])


# --- 2. DATA LOADING & PREP ---
print(">>> ğŸ�—ï¸� Building Heterogeneous Stack (Paper PMC10107388)...")

train = pd.read_csv(CONFIG['TRAIN_PATH'])
test = pd.read_csv(CONFIG['TEST_PATH'])
submission_id = test['id']

if 'id' in train.columns: train = train.drop(columns=['id'])
if 'id' in test.columns: test = test.drop(columns=['id'])

df_all = pd.concat([train.drop(columns=[CONFIG['TARGET']]), test], axis=0).reset_index(drop=True)
df_all.columns = df_all.columns.str.lower().str.strip()

# --- RE-CREATE FEATURES (Your Golden Set) ---
col_age = 'age'; col_bmi = 'bmi'; col_waist = 'waist_to_hip_ratio'
col_bp = 'systolic_bp'; col_activity = 'physical_activity_minutes_per_week'
col_diet = 'diet_score'; col_screen = 'screen_time_hours_per_day'

if col_bmi in df_all.columns and col_age in df_all.columns:
    df_all['Age_x_BMI'] = df_all[col_age] * df_all[col_bmi]
if col_bp in df_all.columns and col_age in df_all.columns:
    df_all['Age_x_BP'] = df_all[col_age] * df_all[col_bp]
if col_bmi in df_all.columns and col_waist in df_all.columns:
    df_all['Visceral_Fat_Proxy'] = df_all[col_bmi] * df_all[col_waist]
if col_screen in df_all.columns and col_activity in df_all.columns:
    df_all['Sedentary_Index'] = df_all[col_screen] / (df_all[col_activity] + 1)
if col_diet in df_all.columns and col_activity in df_all.columns:
    df_all['Healthy_Lifestyle_Score'] = df_all[col_diet] + (df_all[col_activity] / 10)

# Encoding
cat_cols = df_all.select_dtypes(include=['object']).columns.tolist()
if cat_cols:
    le = LabelEncoder()
    for col in cat_cols:
        df_all[col] = df_all[col].astype(str)
        df_all[col] = le.fit_transform(df_all[col])

# Filter Features
X = df_all.iloc[:len(train)][CONFIG['FEATURES']].copy()
X_test = df_all.iloc[len(train):][CONFIG['FEATURES']].copy()
y = train[CONFIG['TARGET']]

# --- SCALING (CRITICAL FOR NEURAL NETS) ---

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# --- 3. LEVEL 1: BASE MODELS ---
print(">>> âš”ï¸� Training Level 1 Models...")
kf = StratifiedKFold(n_splits=CONFIG['FOLDS'], shuffle=True, random_state=CONFIG['SEED'])

oof_lgb = np.zeros(len(X))
oof_xgb = np.zeros(len(X))
oof_nn = np.zeros(len(X))

test_lgb = np.zeros(len(X_test))
test_xgb = np.zeros(len(X_test))
test_nn = np.zeros(len(X_test))

# A. LightGBM (The Tree Specialist)
lgb_params = {
    'objective': 'binary', 'metric': 'auc', 'n_estimators': 3000, 
    'learning_rate': 0.01, 'max_depth': 5, 'num_leaves': 32,
    'subsample': 0.7, 'colsample_bytree': 0.7, 'n_jobs': -1, 'verbose': -1,
    'device': 'gpu'
}

# B. XGBoost (The Other Tree Specialist)
xgb_params = {
    'objective': 'binary:logistic', 'eval_metric': 'auc', 'n_estimators': 3000, 
    'learning_rate': 0.01, 'max_depth': 5, 'subsample': 0.7, 'colsample_bytree': 0.7,
    'tree_method': 'gpu_hist', 'predictor': 'gpu_predictor', 'enable_categorical': False
}

# C. Neural Network (The Diversity Specialist - MLP)
nn_model = MLPClassifier(
    hidden_layer_sizes=(128, 64, 32), activation='relu', solver='adam',
    alpha=0.0001, batch_size=256, learning_rate_init=0.001, max_iter=200,
    early_stopping=True, validation_fraction=0.1, random_state=CONFIG['SEED']
)

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    # Data Split
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # Scaled Split (For NN)
    X_tr_sc, X_val_sc = X_scaled[train_idx], X_scaled[val_idx]
    
    # 1. Train LGBM
    model_lgb = lgb.LGBMClassifier(**lgb_params)
    model_lgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(100, verbose=False)])
    oof_lgb[val_idx] = model_lgb.predict_proba(X_val)[:, 1]
    test_lgb += model_lgb.predict_proba(X_test)[:, 1] / CONFIG['FOLDS']
    
    # 2. Train XGB
    model_xgb = xgb.XGBClassifier(**xgb_params)
    model_xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], early_stopping_rounds=100, verbose=False)
    oof_xgb[val_idx] = model_xgb.predict_proba(X_val)[:, 1]
    test_xgb += model_xgb.predict_proba(X_test)[:, 1] / CONFIG['FOLDS']
    
    # 3. Train Neural Net
    nn_model.fit(X_tr_sc, y_tr)
    oof_nn[val_idx] = nn_model.predict_proba(X_val_sc)[:, 1]
    test_nn += nn_model.predict_proba(X_test_scaled)[:, 1] / CONFIG['FOLDS']
    
    print(f"    Fold {fold+1} Models Trained.")

print(">>> âœ… Level 1 Complete.")


# --- 4. LEVEL 2: STACKING ---
print(">>> ğŸ§  Training Meta-Learner (Logistic Regression)...")

# Input to Meta-Learner
X_stack_train = np.column_stack((oof_lgb, oof_xgb, oof_nn))
X_stack_test = np.column_stack((test_lgb, test_xgb, test_nn))

# Meta-Learner: Logistic Regression
meta_model = LogisticRegression(penalty='l2', C=1.0, random_state=CONFIG['SEED'])
meta_model.fit(X_stack_train, y)

# Final Predictions
final_preds = meta_model.predict_proba(X_stack_test)[:, 1]

# Check the weights (Interpretation)
print(f"    Meta-Learner Weights: {meta_model.coef_[0]}")
print(f"    (LGBM: {meta_model.coef_[0][0]:.2f}, XGB: {meta_model.coef_[0][1]:.2f}, NN: {meta_model.coef_[0][2]:.2f})")

# SAVE
submission = pd.DataFrame({'id': submission_id, 'diagnosed_diabetes': final_preds})
submission.to_csv('submission_paper_stack.csv', index=False)
print(">>> âœ… Stacking Submission Saved: submission_paper_stack.csv")

