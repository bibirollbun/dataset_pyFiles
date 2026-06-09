import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore')

# --- 1. Load Data ---
# Keep duplicates (confirmed necessary for distribution matching)
train = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')

# Drop ID
train = train.drop(columns=['id'])
test_ids = test['id']
test = test.drop(columns=['id'])

# --- 2. Medical Logic Correction (Zero -> NaN) ---
# Only apply to columns where 0 is biologically impossible
impossible_zeros = [
    'glucose', 'cholesterol_total', 'hdl_cholesterol', 'ldl_cholesterol', 
    'triglycerides', 'systolic_bp', 'diastolic_bp', 'bmi', 'waist_to_hip_ratio'
]

print("Cleaning Impossible Zeros...")
for col in impossible_zeros:
    if col in train.columns:
        # Replace 0 with NaN
        train[col] = train[col].replace(0, np.nan)
        test[col] = test[col].replace(0, np.nan)

# --- 3. Feature Engineering ---
def engineer_features(df):
    df = df.copy()
    
    # Ratios (handling NaNs safely)
    df['Chol_HDL_Ratio'] = df['cholesterol_total'] / df['hdl_cholesterol']
    df['BMI_Waist'] = df['bmi'] * df['waist_to_hip_ratio']
    # Mean Arterial Pressure
    df['MAP'] = (df['systolic_bp'] + (2 * df['diastolic_bp'])) / 3
    
    return df

X = engineer_features(train)
X_test = engineer_features(test)

y = X['diagnosed_diabetes']
X = X.drop(columns=['diagnosed_diabetes'])

# --- 4. Crash-Proof Encoding ---
# Automatically select ALL object columns to prevent "Invalid columns" error
object_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
print(f"Encoding Categoricals: {object_cols}")

for col in object_cols:
    le = LabelEncoder()
    # Convert to string to handle mixed types/NaNs in categoricals
    combined = pd.concat([X[col], X_test[col]], axis=0).astype(str)
    le.fit(combined)
    X[col] = le.transform(X[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))

# --- 5. Robust Ensemble Training ---
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

print("\nStarting Ensemble Loop...")

for fold, (trn_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, y_train = X.iloc[trn_idx], y.iloc[trn_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    
    # --- Model A: XGBoost (Depth 6, NaN aware) ---
    xgb_clf = xgb.XGBClassifier(
        n_estimators=1500,
        learning_rate=0.02,
        max_depth=6,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_lambda=2.0,
        tree_method='hist', # Faster and handles NaNs natively
        random_state=42 + fold,
        n_jobs=-1,
        early_stopping_rounds=100
    )
    xgb_clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    p_xgb = xgb_clf.predict_proba(X_val)[:, 1]
    t_xgb = xgb_clf.predict_proba(X_test)[:, 1]

    # --- Model B: LightGBM (Leaf-wise growth) ---
    lgb_clf = lgb.LGBMClassifier(
        n_estimators=1500,
        learning_rate=0.02,
        num_leaves=31,
        subsample=0.7,
        colsample_bytree=0.7,
        random_state=42 + fold,
        n_jobs=-1,
        verbose=-1
    )
    lgb_clf.fit(X_train, y_train)
    p_lgb = lgb_clf.predict_proba(X_val)[:, 1]
    t_lgb = lgb_clf.predict_proba(X_test)[:, 1]
    
    # --- Model C: CatBoost (Symmetric Trees) ---
    cb_clf = cb.CatBoostClassifier(
        iterations=1500,
        learning_rate=0.02,
        depth=6,
        l2_leaf_reg=5,
        random_seed=42 + fold,
        verbose=False,
        allow_writing_files=False
    )
    cb_clf.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=100)
    p_cb = cb_clf.predict_proba(X_val)[:, 1]
    t_cb = cb_clf.predict_proba(X_test)[:, 1]
    
    # --- Blending ---
    # Equal weights act as regularization against any single model's bias
    oof_preds[val_idx] = (p_xgb + p_lgb + p_cb) / 3
    test_preds += ((t_xgb + t_lgb + t_cb) / 3) / 10
    
    # Check individual Fold Score
    print(f"Fold {fold+1} AUC: {roc_auc_score(y_val, oof_preds[val_idx]):.5f}")

# Final Score
print(f"\nOverall CV AUC: {roc_auc_score(y, oof_preds):.5f}")

# --- 6. Submission ---
submission = pd.DataFrame({'id': test_ids, 'diagnosed_diabetes': test_preds})
submission.to_csv('submission.csv', index=False)
print("Submission saved.")

