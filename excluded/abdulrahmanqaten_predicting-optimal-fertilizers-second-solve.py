import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ===================================================================
# ===== Final Strategy: Blending LGBM + XGB on Simple Features ======
# ===================================================================

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import gc

# --- 1. Configuration ---
class CFG:
    TRAIN_PATH = "/kaggle/input/playground-series-s5e6/train.csv"
    TEST_PATH = "/kaggle/input/playground-series-s5e6/test.csv"
    TARGET_COL = 'Fertilizer Name'
    ID_COL = 'id'
    SEED = 42
    N_JOBS = -1

# --- 2. Load and Prepare Data ---
print("Loading original data...")
try:
    train_df = pd.read_csv(CFG.TRAIN_PATH)
    test_df = pd.read_csv(CFG.TEST_PATH)
except FileNotFoundError:
    print("Error: Original data files not found. Please check paths in CFG.")
    exit()

print("Preparing data for modeling...")
X = train_df.drop(columns=[CFG.TARGET_COL, CFG.ID_COL])
y_raw = train_df[CFG.TARGET_COL]
X_test = test_df.drop(columns=[CFG.ID_COL])
test_ids = test_df[CFG.ID_COL]

# Encode categorical features
categorical_features = ['Soil Type', 'Crop Type']
for col in categorical_features:
    le = LabelEncoder()
    combined_series = pd.concat([X[col], X_test[col]]).astype(str)
    le.fit(combined_series)
    X[col] = le.transform(X[col])
    X_test[col] = le.transform(X_test[col])

unique_labels = y_raw.unique()
print(f"Found {len(unique_labels)} unique labels to model.")

# --- 3. Train LightGBM Models ---
lgb_test_preds = {}
lgb_params = {
    'objective': 'binary', 'metric': 'auc', 'n_estimators': 1200,
    'learning_rate': 0.02, 'feature_fraction': 0.7, 'bagging_fraction': 0.7,
    'bagging_freq': 1, 'lambda_l1': 0.1, 'lambda_l2': 0.1,
    'num_leaves': 20, 'verbose': -1, 'n_jobs': CFG.N_JOBS, 'seed': CFG.SEED
}

print("\n--- Phase 1: Training LightGBM Models ---")
for label in unique_labels:
    print(f"Training LGBM model for: {label}")
    y_binary = (y_raw == label).astype(int)
    model = lgb.LGBMClassifier(**lgb_params)
    model.fit(X, y_binary)
    lgb_test_preds[label] = model.predict_proba(X_test)[:, 1]
    gc.collect()

# --- 4. Train XGBoost Models ---
xgb_test_preds = {}
xgb_params = {
    'objective': 'binary:logistic', 'eval_metric': 'auc', 'n_estimators': 1200,
    'learning_rate': 0.02, 'max_depth': 4, 'subsample': 0.7,
    'colsample_bytree': 0.7, 'use_label_encoder': False, 'seed': CFG.SEED, 'n_jobs': CFG.N_JOBS
}

print("\n--- Phase 2: Training XGBoost Models ---")
for label in unique_labels:
    print(f"Training XGBoost model for: {label}")
    y_binary = (y_raw == label).astype(int)
    model = xgb.XGBClassifier(**xgb_params)
    model.fit(X, y_binary)
    xgb_test_preds[label] = model.predict_proba(X_test)[:, 1]
    gc.collect()

print("\nAll models trained successfully!")

# --- 5. Blend Predictions and Generate Submission ---
print("\nBlending predictions and generating final submission...")
lgb_preds_df = pd.DataFrame(lgb_test_preds)
xgb_preds_df = pd.DataFrame(xgb_test_preds)

# Average the probabilities from both models
blended_preds_df = (lgb_preds_df + xgb_preds_df) / 2.0

# Apply the "Top 3" strategy to the blended probabilities
predictions = []
for index, row in blended_preds_df.iterrows():
    top_3_labels = row.sort_values(ascending=False).head(3).index.tolist()
    predictions.append(" ".join(top_3_labels))

# Create the final submission DataFrame
submission_df = pd.DataFrame({CFG.ID_COL: test_ids, CFG.TARGET_COL: predictions})
submission_df.to_csv('submission_lgbm_xgb_blend.csv', index=False)

print("\n'submission_lgbm_xgb_blend.csv' created successfully!")
print("This is our most powerful attempt yet. Good luck with the submission!")
print("\nTop 5 rows of the submission file:")
print(submission_df.head())




