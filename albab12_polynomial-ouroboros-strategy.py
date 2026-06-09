import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder, QuantileTransformer
import warnings

warnings.filterwarnings('ignore')

# --- 1. Load Data ---
train = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')

# Drop ID
train = train.drop(columns=['id'])
test_ids = test['id']
test = test.drop(columns=['id'])

# --- 2. Safe Feature Engineering ---
# We define features to log-transform. We check existence first to be crash-proof.
skewed_candidates = [
    'triglycerides', 'bmi', 'cholesterol_total', 'hdl_cholesterol', 
    'ldl_cholesterol', 'waist_to_hip_ratio'
]

# Only pick columns that actually exist
skewed_cols = [c for c in skewed_candidates if c in train.columns]
print(f"Log-transforming: {skewed_cols}")

for col in skewed_cols:
    # Log(1+x) handles zeros safely
    train[col] = np.log1p(train[col])
    test[col] = np.log1p(test[col])

# --- 3. Robust Encoding (Crash-Proof) ---
# Automatically find all text columns
cat_cols = train.select_dtypes(include=['object', 'category']).columns.tolist()
print(f"Encoding Categoricals: {cat_cols}")

for col in cat_cols:
    le = LabelEncoder()
    # Fill NaN with "MISSING" to give it a category
    train[col] = train[col].fillna("MISSING").astype(str)
    test[col] = test[col].fillna("MISSING").astype(str)
    
    combined = pd.concat([train[col], test[col]], axis=0)
    le.fit(combined)
    train[col] = le.transform(train[col])
    test[col] = le.transform(test[col])

# --- 4. Quantile Transformation (Gaussian) ---
# DART works best when input features are somewhat normal
# We apply this to ALL numeric columns (excluding target)
numeric_cols = train.select_dtypes(include=['float64', 'int64']).columns.tolist()
if 'diagnosed_diabetes' in numeric_cols:
    numeric_cols.remove('diagnosed_diabetes')

print("Applying QuantileTransformer...")
qt = QuantileTransformer(output_distribution='normal', random_state=42)
train[numeric_cols] = qt.fit_transform(train[numeric_cols])
test[numeric_cols] = qt.transform(test[numeric_cols])

X = train.drop(columns=['diagnosed_diabetes'])
y = train['diagnosed_diabetes']
X_test = test

# --- 5. The DART Strategy ---
# DART is much slower than 'hist', but it generalizes better.
# rate_drop=0.1 means 10% of trees are dropped in each round.

print("\nStarting DART Training...")
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

oof_preds_dart = np.zeros(len(X))
test_preds_dart = np.zeros(len(X_test))

# We also run a HistGradientBoosting as a "Safety Net"
oof_preds_hgb = np.zeros(len(X))
test_preds_hgb = np.zeros(len(X_test))

for fold, (trn_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, y_train = X.iloc[trn_idx], y.iloc[trn_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    
    # Model 1: XGBoost DART
    # Note: enable_categorical=False because we manually encoded above
    dart_clf = xgb.XGBClassifier(
        n_estimators=1000,          # DART needs fewer trees or it gets too slow
        learning_rate=0.05,         # Higher LR is okay for DART
        booster='dart',             # <--- THE KEY CHANGE
        rate_drop=0.1,              # Drop 10% of trees
        skip_drop=0.5,              # 50% chance to skip dropout (hybrid mode)
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        n_jobs=-1,
        tree_method='hist'          # Use histogram binning for speed
    )
    
    dart_clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    oof_preds_dart[val_idx] = dart_clf.predict_proba(X_val)[:, 1]
    test_preds_dart += dart_clf.predict_proba(X_test)[:, 1] / 10
    
    # Model 2: HistGradientBoosting (Standard Baseline)
    hgb_clf = HistGradientBoostingClassifier(
        max_iter=1000,
        learning_rate=0.03,
        max_depth=6,
        l2_regularization=3.0,
        random_state=42 + fold
    )
    hgb_clf.fit(X_train, y_train)
    
    oof_preds_hgb[val_idx] = hgb_clf.predict_proba(X_val)[:, 1]
    test_preds_hgb += hgb_clf.predict_proba(X_test)[:, 1] / 10
    
    print(f"Fold {fold+1} complete.")

# --- 6. Blending ---
# DART is high variance, high potential. HGB is stable.
# We blend them 60/40.

avg_oof = (0.6 * oof_preds_dart) + (0.4 * oof_preds_hgb)
print(f"\nDART + HGB CV AUC: {roc_auc_score(y, avg_oof):.5f}")

avg_test = (0.6 * test_preds_dart) + (0.4 * test_preds_hgb)

# --- Submission ---
submission = pd.DataFrame({'id': test_ids, 'diagnosed_diabetes': avg_test})
submission.to_csv('submission.csv', index=False)
print("Submission saved.")

