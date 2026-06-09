import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
import warnings

warnings.filterwarnings('ignore')

print("ğŸš€ STARTING PHASE 10: DYNAMIC SNIPER (Top 1% Version)")
print("="*60)

# ============================================================================
# 1. CONFIGURATION
# ============================================================================
TRAIN_PATH = '/kaggle/input/detecting-reversal-points-in-us-equities/new_comptetition_data/train.csv'
TEST_PATH = '/kaggle/input/detecting-reversal-points-in-us-equities/new_comptetition_data/test.csv'

# ADJUSTMENT: Target Top 1% (approx 10-12 trades) instead of Top 0.3%
# This ensures we don't miss obvious opportunities while staying safe.
TARGET_PERCENTILE = 99.0 

GOLDEN_FEATURES = [
    'ratio', 'sm_momentum', 'sm_ratio', 'momentum', 
    'occurs_within_zone_100.0_zone_98.5_10', 
    'trending_down_and_above_100.0', 'zone_99.0'
]

# ============================================================================
# 2. DATA LOADING & ENGINEERING
# ============================================================================
def load_and_engineer(path):
    print(f"âš™ï¸� Processing {path.split('/')[-1]}...")
    header = pd.read_csv(path, nrows=0)
    cols = [c for c in GOLDEN_FEATURES if c in header.columns]
    cols += [c for c in ['id', 'ticker_id', 'class_label'] if c in header.columns]
    
    df = pd.read_csv(path, usecols=cols, low_memory=False)
    
    # Context Engineering
    groups = df.groupby('ticker_id')
    for col in ['momentum', 'ratio', 'sm_momentum']:
        if col in df.columns:
            roll_mean = groups[col].transform(lambda x: x.rolling(20, min_periods=1).mean())
            roll_std  = groups[col].transform(lambda x: x.rolling(20, min_periods=1).std())
            df[f'{col}_ZSCORE'] = (df[col] - roll_mean) / (roll_std + 1e-6)

    # DIVERGENCE (The "Search" Feature)
    if 'momentum' in df.columns and 'ratio' in df.columns:
        df['MOM_RATIO_DIV'] = df['momentum'] / (df['ratio'] + 1e-6)

    return df.fillna(0)

train = load_and_engineer(TRAIN_PATH)
test = load_and_engineer(TEST_PATH)

# ============================================================================
# 3. TRAINING
# ============================================================================
print("\nğŸ�‹ï¸� Training Ensemble...")
exclude = ['id', 'ticker_id', 'class_label', 'target']
features = [c for c in train.columns if c not in exclude and pd.api.types.is_numeric_dtype(train[c])]

X = train[features]
y = train['class_label'].map({'None': 0, 'H': 1, 'L': 2}).fillna(0).astype(int)
X_test = test[features]

models = []
tscv = TimeSeriesSplit(n_splits=5)

for fold, (train_idx, val_idx) in enumerate(tscv.split(X, y)):
    X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    
    # Increased num_leaves slightly as per search suggestion to capture complexity
    model = lgb.LGBMClassifier(
        n_estimators=1000, learning_rate=0.02, num_leaves=31, max_depth=6,
        class_weight='balanced', reg_alpha=1.0, reg_lambda=1.0,
        random_state=42 + fold, verbose=-1, n_jobs=-1
    )
    model.fit(X_tr, y_tr)
    models.append(model)

# ============================================================================
# 4. DYNAMIC SNIPER LOGIC
# ============================================================================
print("\nğŸ”® Calculating Dynamic Thresholds (Top 1.0%)...")

test_probs = np.zeros((len(X_test), 3))
for model in models:
    test_probs += model.predict_proba(X_test)
test_probs /= len(models)

# Determine Thresholds based on Test Set Distribution
thresh_h_dynamic = np.percentile(test_probs[:, 1], TARGET_PERCENTILE)
thresh_l_dynamic = np.percentile(test_probs[:, 2], TARGET_PERCENTILE)

# Safety Floor (Keep Intelligence High) - Maintained 0.55
thresh_h = max(thresh_h_dynamic, 0.55)
thresh_l = max(thresh_l_dynamic, 0.55)

print(f"   H Threshold: {thresh_h:.4f}")
print(f"   L Threshold: {thresh_l:.4f}")

final_preds = []
stats = {'None': 0, 'H': 0, 'L': 0}

for probs in test_probs:
    p_none, p_h, p_l = probs
    
    if p_h > thresh_h and p_h > p_l:
        final_preds.append('H')
        stats['H'] += 1
    elif p_l > thresh_l and p_l > p_h:
        final_preds.append('L')
        stats['L'] += 1
    else:
        final_preds.append('None')
        stats['None'] += 1

# ============================================================================
# 5. SAVE
# ============================================================================
sub = pd.DataFrame({'id': test['id'], 'class_label': final_preds})
sub.to_csv('submission.csv', index=False)

print("\n" + "="*60)
print(f"âœ… SUBMISSION READY")
print(f"ğŸ“Š Signal Distribution: {stats}")
print("="*60)




