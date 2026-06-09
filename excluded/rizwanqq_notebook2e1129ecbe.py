# =====================================
# 1. Install / Import Libraries
# =====================================
!pip install lightgbm catboost xgboost --quiet

import numpy as np
import pandas as pd
import lightgbm as lgb
import catboost as cat
import xgboost as xgb
import gc  # for garbage collection (useful with large datasets)
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score

# =====================================
# 2. Load Main Data
# =====================================
train_df = pd.read_csv("/kaggle/input/home-credit-default-risk/application_train.csv")
test_df  = pd.read_csv("/kaggle/input/home-credit-default-risk/application_test.csv")

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)

# Save test IDs for submission
test_ids = test_df["SK_ID_CURR"]

# Extract TARGET and drop from train
y = train_df["TARGET"].copy()
train_df.drop(columns=["TARGET"], inplace=True)

# =====================================
# 3. Load Auxiliary Datasets
# =====================================
bureau = pd.read_csv("/kaggle/input/home-credit-default-risk/bureau.csv")
bureau_balance = pd.read_csv("/kaggle/input/home-credit-default-risk/bureau_balance.csv")
pos_cash = pd.read_csv("/kaggle/input/home-credit-default-risk/POS_CASH_balance.csv")
credit_card = pd.read_csv("/kaggle/input/home-credit-default-risk/credit_card_balance.csv")
installments = pd.read_csv("/kaggle/input/home-credit-default-risk/installments_payments.csv")

# =====================================
# 4. Basic Aggregations / Feature Engineering
# =====================================

def aggregate_data(df, group_key):
    """ Aggregates numerical features. """
    num_agg = df.select_dtypes(include=['number']).groupby(group_key).agg(['mean', 'sum', 'max', 'min'])
    num_agg.columns = [f"{col[0]}_{col[1]}" for col in num_agg.columns]
    return num_agg.reset_index()

# Aggregate datasets
bureau_agg = aggregate_data(bureau, "SK_ID_CURR")
pos_agg = aggregate_data(pos_cash, "SK_ID_CURR")
credit_agg = aggregate_data(credit_card, "SK_ID_CURR")
installments_agg = aggregate_data(installments, "SK_ID_CURR")

# Merge into train & test
for dataset in [bureau_agg, pos_agg, credit_agg, installments_agg]:
    train_df = train_df.merge(dataset, on="SK_ID_CURR", how="left")
    test_df  = test_df.merge(dataset, on="SK_ID_CURR", how="left")

del bureau, pos_cash, credit_card, installments, bureau_agg, pos_agg, credit_agg, installments_agg
gc.collect()

# =====================================
# 5. Handle Object Columns (Label Encoding)
# =====================================

train_len = len(train_df)
df_full = pd.concat([train_df, test_df], axis=0, ignore_index=True)

# Identify object columns
obj_cols = df_full.select_dtypes(include=["object"]).columns

# Label encode categorical features
for col in obj_cols:
    df_full[col], _ = pd.factorize(df_full[col], sort=True)

train_df = df_full.iloc[:train_len, :].copy()
test_df  = df_full.iloc[train_len:, :].copy()

del df_full
gc.collect()

# =====================================
# 6. Final Preprocessing Steps
# =====================================

# Fill numeric NaNs with 0
for col in train_df.columns:
    if str(train_df[col].dtype) in ["float64", "int64"]:
        train_df[col].fillna(0, inplace=True)

for col in test_df.columns:
    if str(test_df[col].dtype) in ["float64", "int64"]:
        test_df[col].fillna(0, inplace=True)

# Drop ID columns
drop_cols = ["SK_ID_CURR"]
X = train_df.drop(columns=drop_cols, errors="ignore")
X_test = test_df.drop(columns=drop_cols, errors="ignore")

print("X shape:", X.shape, "| y shape:", y.shape)
print("X_test shape:", X_test.shape)

# =====================================
# 7. Train Base Models (GPU Enabled)
# =====================================

# 7.1 CatBoost Model
print("Training CatBoost model...")
cat_model = cat.CatBoostClassifier(
    iterations=6000, learning_rate=0.01, depth=8, l2_leaf_reg=3, 
    eval_metric="AUC", random_seed=42, task_type="GPU", verbose=500
)
cat_model.fit(X, y)

# 7.2 LightGBM Model
print("Training LightGBM model...")
lgb_model = lgb.LGBMClassifier(
    n_estimators=6000, learning_rate=0.01, num_leaves=32, max_depth=8,
    feature_fraction=0.8, subsample=0.8, random_state=42
)
lgb_model.fit(X, y)

# 7.3 XGBoost Model
print("Training XGBoost model...")
xgb_model = xgb.XGBClassifier(
    n_estimators=6000, learning_rate=0.01, max_depth=8, subsample=0.8, 
    colsample_bytree=0.8, gamma=0.02, reg_alpha=0.02, reg_lambda=0.02,
    tree_method="gpu_hist", verbosity=0, random_state=42
)
xgb_model.fit(X, y)

# =====================================
# 8. Stacked Model (Kitchen Sink)
# =====================================
print("Stacking Models using Ridge Regression...")

# Predictions from base models
y_pred_cat = cat_model.predict_proba(X)[:, 1]
y_pred_lgb = lgb_model.predict_proba(X)[:, 1]
y_pred_xgb = xgb_model.predict_proba(X)[:, 1]

# Train Meta-Model (Ridge Regression)
meta_model = Ridge(alpha=0.1)
stacked_train = np.column_stack((y_pred_cat, y_pred_lgb, y_pred_xgb))
meta_model.fit(stacked_train, y)

# Final Predictions (Test Set)
stacked_test = np.column_stack((
    cat_model.predict_proba(X_test)[:, 1],
    lgb_model.predict_proba(X_test)[:, 1],
    xgb_model.predict_proba(X_test)[:, 1]
))
y_pred_test = meta_model.predict(stacked_test)

# Ensure valid probability range (0 to 1)
y_pred_test = np.clip(y_pred_test, 0, 1)

# =====================================
# 9. Create Submission File (Matching Kaggle Sample Format)
# =====================================
submission = pd.DataFrame({
    "SK_ID_CURR": test_ids,
    "TARGET": y_pred_test
})

submission.to_csv("submission_stacked1.csv", index=False)
print("✅ Submission file 'submission_stacked.csv' created! Upload it to Kaggle for scoring.")





