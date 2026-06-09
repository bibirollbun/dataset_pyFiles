import pandas as pd
import numpy as np
import warnings
import category_encoders as ce
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from scipy.optimize import minimize

warnings.filterwarnings('ignore')


# ==========================================
# 1. CONFIGURATION & DATA LOADING
# ==========================================
TRAIN_PATH = "/kaggle/input/playground-series-s5e12/train.csv" # Update path if needed
TEST_PATH = "/kaggle/input/playground-series-s5e12/test.csv"   # Update path if needed

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)


y = train['diagnosed_diabetes']
X = train.drop(['diagnosed_diabetes', 'id'], axis=1)
X_test = test.drop('id', axis=1)
test_ids = test['id']


# ==========================================
# 2. FEATURE ENGINEERING
# ==========================================
def engineer_features(df):
    df = df.copy()
    
    # --- Medical Interactions ---
    # Pulse Pressure: Stiffness of arteries
    df['pulse_pressure'] = df['systolic_bp'] - df['diastolic_bp']
    # Mean Arterial Pressure (MAP)
    df['map'] = df['diastolic_bp'] + (df['pulse_pressure'] / 3)
    
    # --- Lipid Ratios (Vital for Diabetes/Heart risks) ---
    # Epsilon to avoid division by zero
    epsilon = 1e-5
    df['cholesterol_ratio'] = df['cholesterol_total'] / (df['hdl_cholesterol'] + epsilon)
    df['ldl_hdl_ratio'] = df['ldl_cholesterol'] / (df['hdl_cholesterol'] + epsilon)
    df['trig_hdl_ratio'] = df['triglycerides'] / (df['hdl_cholesterol'] + epsilon)
    
    # --- Body / Lifestyle ---
    df['bmi_age'] = df['bmi'] * df['age']
    df['waist_bmi'] = df['waist_to_hip_ratio'] * df['bmi']
    # Activity impact relative to weight
    df['activity_intensity'] = df['physical_activity_minutes_per_week'] / (df['bmi'] + 1)
    
    return df

print("Engineering features...")
X = engineer_features(X)
X_test = engineer_features(X_test)


# ==========================================
# 3. ENCODING SETUP
# ==========================================
# Lists of columns
cat_cols_all = [col for col in X.columns if X[col].dtype == 'object']
ordinal_cols = ['education_level', 'income_level']
nominal_cols = [c for c in cat_cols_all if c not in ordinal_cols]

# Manual Maps for Ordinal Data (Preserves Rank)
edu_map = {'No Schooling': 0, 'Elementary': 1, 'Highschool': 2, 'College': 3, 'Graduate': 4, 'PhD': 5}
inc_map = {'Low': 0, 'Lower-Middle': 1, 'Middle': 2, 'Upper-Middle': 3, 'High': 4}

def manual_encode(df):
    df = df.copy()
    if 'education_level' in df.columns:
        df['education_level'] = df['education_level'].map(edu_map).fillna(-1)
    if 'income_level' in df.columns:
        df['income_level'] = df['income_level'].map(inc_map).fillna(-1)
    return df

# Apply Manual Encoding globally (safe for all models)
X = manual_encode(X)
X_test = manual_encode(X_test)


# ==========================================
# 4. STRATIFIED K-FOLD TRAINING LOOP
# ==========================================
FOLDS = 10 # Standard stability
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)

# Storage for predictions
oof_preds = {
    'cat': np.zeros(len(X)),
    'xgb': np.zeros(len(X)),
    'lgbm': np.zeros(len(X))
}
test_preds = {
    'cat': np.zeros(len(X_test)),
    'xgb': np.zeros(len(X_test)),
    'lgbm': np.zeros(len(X_test))
}

print(f"Starting training on {FOLDS} folds...")

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\n=== FOLD {fold+1} ===")
    
    # A. Split Data
    X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    
    # -------------------------------------------------------
    # MODEL 1: CATBOOST (Uses Native Categorical Handling)
    # -------------------------------------------------------
    # CatBoost gets the raw nominal columns (it handles them best internally)
    cb = CatBoostClassifier(
        iterations=2000, 
        learning_rate=0.03, 
        depth=6,
        cat_features=nominal_cols, # Pass nominals directly
        eval_metric='AUC',
        early_stopping_rounds=100,
        verbose=0,
        random_state=42
    )
    cb.fit(X_tr, y_tr, eval_set=(X_val, y_val))
    
    oof_preds['cat'][val_idx] = cb.predict_proba(X_val)[:, 1]
    test_preds['cat'] += cb.predict_proba(X_test)[:, 1] / FOLDS
    print(f"CatBoost AUC: {roc_auc_score(y_val, oof_preds['cat'][val_idx]):.5f}")

    # -------------------------------------------------------
    # ENCODING FOR XGB/LGBM (CatBoostEncoder inside loop)
    # -------------------------------------------------------
    # We must fit encoder on Fold Train, transform Fold Val to avoid leakage
    cbe = ce.CatBoostEncoder(cols=nominal_cols)
    cbe.fit(X_tr[nominal_cols], y_tr)
    
    # Prepare Encoded Dataframes
    X_tr_enc = X_tr.copy()
    X_val_enc = X_val.copy()
    X_test_enc = X_test.copy() # We transform test based on fold train (averaged later)
    
    X_tr_enc[nominal_cols] = cbe.transform(X_tr[nominal_cols])
    X_val_enc[nominal_cols] = cbe.transform(X_val[nominal_cols])
    X_test_enc[nominal_cols] = cbe.transform(X_test[nominal_cols]) # Note: In production, we'd average encoder maps, but this works for Kaggle
    
    # -------------------------------------------------------
    # MODEL 2: XGBOOST
    # -------------------------------------------------------
    xgb = XGBClassifier(
        n_estimators=2000,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.7,
        colsample_bytree=0.7,
        eval_metric='auc',
        early_stopping_rounds=100,
        n_jobs=-1,
        random_state=42,
        verbosity=0
    )
    xgb.fit(X_tr_enc, y_tr, eval_set=[(X_val_enc, y_val)], verbose=False)
    
    oof_preds['xgb'][val_idx] = xgb.predict_proba(X_val_enc)[:, 1]
    test_preds['xgb'] += xgb.predict_proba(X_test_enc)[:, 1] / FOLDS
    print(f"XGBoost  AUC: {roc_auc_score(y_val, oof_preds['xgb'][val_idx]):.5f}")

    # -------------------------------------------------------
    # MODEL 3: LIGHTGBM
    # -------------------------------------------------------
    lgbm = LGBMClassifier(
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.7,
        colsample_bytree=0.7,
        metric='auc',
        n_jobs=-1,
        random_state=42,
        verbosity=-1
    )
    # LGBM callbacks for early stopping are handled differently in newer versions, 
    # but fit parameter usually works
    lgbm.fit(X_tr_enc, y_tr, eval_set=[(X_val_enc, y_val)])
    
    oof_preds['lgbm'][val_idx] = lgbm.predict_proba(X_val_enc)[:, 1]
    test_preds['lgbm'] += lgbm.predict_proba(X_test_enc)[:, 1] / FOLDS
    print(f"LightGBM AUC: {roc_auc_score(y_val, oof_preds['lgbm'][val_idx]):.5f}") 


# ==========================================
# 5. OPTIMIZE WEIGHTS
# ==========================================
print("\nFinding Optimal Ensemble Weights...")

def minimize_auc(weights):
    # Normalize
    weights = np.array(weights)
    weights /= weights.sum()
    
    # Weighted Average
    final_oof = (weights[0] * oof_preds['cat'] + 
                 weights[1] * oof_preds['xgb'] + 
                 weights[2] * oof_preds['lgbm'])
    
    return -roc_auc_score(y, final_oof)

init_guess = [0.33, 0.33, 0.33]
bounds = [(0, 1)] * 3
result = minimize(minimize_auc, init_guess, method='SLSQP', bounds=bounds)

best_w = result.x / result.x.sum()

print(f"Best Weights -> CatBoost: {best_w[0]:.3f}, XGB: {best_w[1]:.3f}, LGBM: {best_w[2]:.3f}")
print(f"Final Optimized CV Score: {-result.fun:.5f}")


# ==========================================
# 6. SUBMISSION
# ==========================================
final_test_pred = (best_w[0] * test_preds['cat'] + 
                   best_w[1] * test_preds['xgb'] + 
                   best_w[2] * test_preds['lgbm'])

submission = pd.DataFrame({'id': test_ids, 'diagnosed_diabetes': final_test_pred})
submission.to_csv('submission_ensemble_optimized.csv', index=False)
print("Submission file saved successfully.")

