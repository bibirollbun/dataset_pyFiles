# =====================================
# 1. Install / Import Libraries
# =====================================
!pip install lightgbm --quiet

import numpy as np
import pandas as pd
import lightgbm as lgb
import gc  # for garbage collection (useful with large datasets)

# =====================================
# 2. Load Main Data
# =====================================
train_df = pd.read_csv("/kaggle/input/home-credit-default-risk/application_train.csv")
test_df  = pd.read_csv("/kaggle/input/home-credit-default-risk/application_test.csv")

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)

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

print("Bureau shape:", bureau.shape)
print("Bureau Balance shape:", bureau_balance.shape)
print("POS_CASH shape:", pos_cash.shape)
print("Credit Card shape:", credit_card.shape)
print("Installments shape:", installments.shape)

# =====================================
# 4. Basic Aggregations / Feature Engineering
# =====================================

# -------------------------------
# 4.1 bureau + bureau_balance
# -------------------------------
# Aggregate bureau_balance at SK_ID_BUREAU level
bureau_balance_agg = bureau_balance.groupby("SK_ID_BUREAU").agg({
    "MONTHS_BALANCE": ["min", "max", "mean"]
})

bureau_balance_agg.columns = ["BB_" + "_".join(col).upper() for col in bureau_balance_agg.columns.ravel()]
bureau_balance_agg.reset_index(inplace=True)

# Merge back to bureau
bureau = bureau.merge(bureau_balance_agg, on="SK_ID_BUREAU", how="left")
del bureau_balance, bureau_balance_agg
gc.collect()

# Now aggregate bureau at SK_ID_CURR level
bureau_agg = bureau.groupby("SK_ID_CURR").agg({
    "DAYS_CREDIT": ["min", "max", "mean"],
    "CREDIT_DAY_OVERDUE": ["max", "mean"],
    "AMT_CREDIT_SUM": ["sum", "mean"],
    "AMT_CREDIT_SUM_DEBT": ["sum", "mean"],
    "AMT_CREDIT_SUM_OVERDUE": ["sum", "mean"],
    "BB_MONTHS_BALANCE_MIN": ["min", "mean"],
    "BB_MONTHS_BALANCE_MAX": ["max", "mean"]
    # Add more if desired
})

bureau_agg.columns = ["BURO_" + "_".join(col).upper() for col in bureau_agg.columns.ravel()]
bureau_agg.reset_index(inplace=True)

# Merge into train & test
train_df = train_df.merge(bureau_agg, on="SK_ID_CURR", how="left")
test_df  = test_df.merge(bureau_agg, on="SK_ID_CURR", how="left")

del bureau, bureau_agg
gc.collect()

# -------------------------------
# 4.2 POS_CASH_balance
# -------------------------------
pos_agg = pos_cash.groupby("SK_ID_CURR").agg({
    "MONTHS_BALANCE": ["min", "max", "mean"],
    "SK_DPD": ["max", "mean"],
    "SK_DPD_DEF": ["max", "mean"]
    # Add more if desired
})

pos_agg.columns = ["POS_" + "_".join(col).upper() for col in pos_agg.columns.ravel()]
pos_agg.reset_index(inplace=True)

train_df = train_df.merge(pos_agg, on="SK_ID_CURR", how="left")
test_df  = test_df.merge(pos_agg, on="SK_ID_CURR", how="left")

del pos_cash, pos_agg
gc.collect()

# -------------------------------
# 4.3 credit_card_balance
# -------------------------------
cc_agg = credit_card.groupby("SK_ID_CURR").agg({
    "MONTHS_BALANCE": ["min", "max", "mean"],
    "AMT_BALANCE": ["sum", "mean", "max"],
    "AMT_CREDIT_LIMIT_ACTUAL": ["mean", "max"],
    "SK_DPD": ["max", "mean"]
    # Add more if desired
})

cc_agg.columns = ["CC_" + "_".join(col).upper() for col in cc_agg.columns.ravel()]
cc_agg.reset_index(inplace=True)

train_df = train_df.merge(cc_agg, on="SK_ID_CURR", how="left")
test_df  = test_df.merge(cc_agg, on="SK_ID_CURR", how="left")

del credit_card, cc_agg
gc.collect()

# -------------------------------
# 4.4 installments_payments
# -------------------------------
install_agg = installments.groupby("SK_ID_CURR").agg({
    "NUM_INSTALMENT_VERSION": ["nunique"],
    "AMT_INSTALMENT": ["sum", "mean", "max"],
    "AMT_PAYMENT": ["sum", "mean", "min"],
    "DAYS_ENTRY_PAYMENT": ["min", "max", "mean"]
    # Add more if desired
})

install_agg.columns = ["INS_" + "_".join(col).upper() for col in install_agg.columns.ravel()]
install_agg.reset_index(inplace=True)

train_df = train_df.merge(install_agg, on="SK_ID_CURR", how="left")
test_df  = test_df.merge(install_agg, on="SK_ID_CURR", how="left")

del installments, install_agg
gc.collect()

# =====================================
# 5. Handle Object Columns (Label Encoding or One-Hot)
# =====================================

# It's often easiest to combine train & test, then encode
train_len = len(train_df)
df_full = pd.concat([train_df, test_df], axis=0, ignore_index=True)

# Identify object columns
obj_cols = df_full.select_dtypes(include=["object"]).columns

# Label-encode each object column
for col in obj_cols:
    df_full[col], _ = pd.factorize(df_full[col], sort=True)

# Split back into train & test
train_df = df_full.iloc[:train_len, :].copy()
test_df  = df_full.iloc[train_len:, :].copy()

del df_full
gc.collect()

# =====================================
# 6. Final Preprocessing Steps
# =====================================

# Fill numeric NaNs with 0 (simple approach)
for col in train_df.columns:
    if str(train_df[col].dtype) in ["float64", "int64"]:
        train_df[col].fillna(0, inplace=True)

for col in test_df.columns:
    if str(test_df[col].dtype) in ["float64", "int64"]:
        test_df[col].fillna(0, inplace=True)

# Drop ID columns or non-feature columns if needed
drop_cols = ["SK_ID_CURR", "SK_ID_BUREAU"]  # Might not exist in the merged data, so ignore errors
X = train_df.drop(columns=drop_cols, errors="ignore")
X_test = test_df.drop(columns=drop_cols, errors="ignore")

print("X shape:", X.shape, "| y shape:", y.shape)
print("X_test shape:", X_test.shape)

# =====================================
# 7. Train LightGBM Model
# =====================================
train_data = lgb.Dataset(X, label=y)

params = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.05,
    "num_leaves": 32,
    "feature_fraction": 0.8,
    "subsample": 0.8,
    "random_state": 42
}

# For simplicity, we're not using a validation set here (not recommended for final).
model = lgb.train(params, train_data, num_boost_round=500)

# =====================================
# 8. Prediction & Submission
# =====================================
y_pred = model.predict(X_test)

submission = pd.DataFrame({
    "SK_ID_CURR": test_df["SK_ID_CURR"],
    "TARGET": y_pred
})

submission.to_csv("submission_lgb.csv", index=False)
print("Submission file 'submission_lgb.csv' created! Upload it to Kaggle for scoring.")


